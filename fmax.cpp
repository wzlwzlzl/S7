#include "kernel_operator.h"
#include <type_traits>
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 3;
constexpr int32_t inputVarNum = 2;
constexpr int32_t maxDimNum = 64;

class TileVar {
public:
  uint32_t CoreDataNum;
  uint32_t finalTileNum;
  uint32_t tileDataNum;
  uint32_t TailDataNum;
  uint64_t input_length;
  uint64_t other_length;
  int64_t numshapes;
  int64_t ss[inputVarNum * maxDimNum];
  int64_t sf[maxDimNum];
};

template <typename TYPE_INPUT, typename TYPE_OTHER, typename TYPE_OUT>
class KernelFmax_Broadcast {
public:
  __aicore__ inline KernelFmax_Broadcast() {}
  __aicore__ inline void Init(GM_ADDR input, GM_ADDR other, GM_ADDR out,
                              TileVar *tilevar, TPipe *pipeIn) {
    this->pipe = pipeIn;
    ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
    this->coreDataNum = tilevar->CoreDataNum;
    this->tileNum64 = tilevar->finalTileNum;
    this->tileDataNum = tilevar->tileDataNum;
    this->tailDataNum = tilevar->TailDataNum;
    this->numshapes = tilevar->numshapes;
    for (int inputNo = 0; inputNo < inputVarNum; ++inputNo) {
      for (int dim = 0; dim < maxDimNum; ++dim) {
        this->shape[inputNo][dim] = tilevar->ss[inputNo * maxDimNum + dim];
      }
    }
    for (int i = 0; i < maxDimNum; ++i) {
      this->shapefull[i] = tilevar->sf[i];
    }
    for (int i = 0; i < maxDimNum; ++i) {
      this->outputStrides[i] = 0;
    }
    if (this->numshapes > 0) {
      const int64_t dimOffset = maxDimNum - this->numshapes;
      uint64_t running = 1ULL;
      for (int i = this->numshapes - 1; i >= 0; --i) {
        this->outputStrides[dimOffset + i] = running;
        int64_t d = this->shapefull[dimOffset + i];
        running *= (d > 0 ? static_cast<uint64_t>(d) : 1ULL);
      }
    }
    for (int inputNo = 0; inputNo < inputVarNum; ++inputNo) {
      for (int i = 0; i < maxDimNum; ++i) {
        this->inputStrides[inputNo][i] = 0;
      }
      if (this->numshapes > 0) {
        const int64_t dimOffset = maxDimNum - this->numshapes;
        uint64_t running = 1ULL;
        for (int i = this->numshapes - 1; i >= 0; --i) {
          this->inputStrides[inputNo][dimOffset + i] = running;
          int64_t d = this->shape[inputNo][dimOffset + i];
          running *= (d > 0 ? static_cast<uint64_t>(d) : 1ULL);
        }
      }
    }
    this->total_length = 0ULL;
    if (this->coreDataNum != 0) {
      uint64_t temp_total = 1ULL;
      if (this->numshapes > 0) {
        const int64_t dimOffset = maxDimNum - this->numshapes;
        for (int i = 0; i < this->numshapes; ++i) {
          int64_t d = this->shapefull[dimOffset + i];
          if (d <= 0) { temp_total = 0; break; }
          temp_total = (uint64_t)((__int128)temp_total * (uint64_t)d);
        }
      }
      this->total_length = temp_total;
    }
    this->input_Length = tilevar->input_length;
    this->other_Length = tilevar->other_length;

    int64_t block_id = GetBlockIdx();
    int64_t num_blocks = GetBlockNum();
    if (num_blocks <= 0) num_blocks = 1;
    uint64_t core_data_base =
        (uint64_t)((__int128)this->total_length / (uint64_t)num_blocks);
    uint64_t rem =
        (uint64_t)((__int128)this->total_length % (uint64_t)num_blocks);
    this->my_core_data =
        core_data_base + (static_cast<uint64_t>(block_id) < rem ? 1ULL : 0ULL);
    uint64_t extra_cores_before =
        (static_cast<uint64_t>(block_id) < rem ? static_cast<uint64_t>(block_id)
                                               : rem);
    this->core_start =
        (uint64_t)((__int128)static_cast<uint64_t>(block_id) * core_data_base) +
        extra_cores_before;

    uint64_t actual_tile_num_64 =
        (this->my_core_data + this->tileDataNum - 1ULL) / this->tileDataNum;
    if (actual_tile_num_64 > UINT32_MAX) {
      this->tileNum64 = UINT32_MAX;
    } else {
      this->tileNum64 = static_cast<uint64_t>(actual_tile_num_64);
    }

    uint32_t core_buffer_size = static_cast<uint32_t>(
        this->my_core_data > UINT32_MAX ? UINT32_MAX : this->my_core_data);

    outGm.SetGlobalBuffer((__gm__ DTYPE_OUT *)out + this->core_start,
                          core_buffer_size);

    bool input_is_broadcast = (this->input_Length != this->total_length);
    uint64_t input_start = input_is_broadcast ? 0ULL : this->core_start;
    uint32_t input_size = input_is_broadcast ?
        static_cast<uint32_t>(this->input_Length > UINT32_MAX ? UINT32_MAX : this->input_Length) :
        core_buffer_size;
    inputGm.SetGlobalBuffer((__gm__ DTYPE_INPUT *)input + input_start, input_size);

    bool other_is_broadcast = (this->other_Length != this->total_length);
    uint64_t other_start = other_is_broadcast ? 0ULL : this->core_start;
    uint32_t other_size = other_is_broadcast ?
        static_cast<uint32_t>(this->other_Length > UINT32_MAX ? UINT32_MAX : this->other_Length) :
        core_buffer_size;
    otherGm.SetGlobalBuffer((__gm__ DTYPE_OTHER *)other + other_start, other_size);

    pipe->InitBuffer(inQueueInput, BUFFER_NUM,
                     this->tileDataNum * sizeof(DTYPE_INPUT));
    pipe->InitBuffer(inQueueOther, BUFFER_NUM,
                     this->tileDataNum * sizeof(DTYPE_OTHER));
    pipe->InitBuffer(outQueueOut, BUFFER_NUM,
                     this->tileDataNum * sizeof(DTYPE_OUT));
    pipe->InitBuffer(tmpBuf1, this->tileDataNum * sizeof(float));
    pipe->InitBuffer(tmpBuf2, this->tileDataNum * sizeof(float));

    this->input_is_scalar = (this->input_Length == 1ULL);
    this->other_is_scalar = (this->other_Length == 1ULL);
    if (this->input_is_scalar) {
      if constexpr (std::is_same_v<DTYPE_INPUT, bool>) {
        this->input_scalar_bool = inputGm.GetValue(0);
      }
    }
    if (this->other_is_scalar) {
      if constexpr (std::is_same_v<DTYPE_OTHER, bool>) {
        this->other_scalar_bool = otherGm.GetValue(0);
      }
    }
  }

  __aicore__ inline void Process() {
    const uint32_t elements_per_32bytes = 32 / sizeof(DTYPE_OUT);
    const uint32_t align_mask = elements_per_32bytes - 1;
    uint32_t actual_tile_num =
        (this->my_core_data + this->tileDataNum - 1) / this->tileDataNum;
    if (actual_tile_num == 0 || this->my_core_data == 0) {
      return;
    }
    {
      uint32_t tile_start0 = 0;
      uint32_t remaining0 = this->my_core_data - tile_start0;
      uint32_t logicalNum0 =
          (remaining0 > this->tileDataNum) ? this->tileDataNum : remaining0;
      uint32_t computeNum0 = (logicalNum0 + align_mask) & ~align_mask;
      if (computeNum0 > this->tileDataNum)
        computeNum0 = this->tileDataNum;
      CopyIn(0, logicalNum0, computeNum0, tile_start0);
    }
    for (uint32_t i = 1; i < actual_tile_num; ++i) {
      uint32_t prev_start = (i - 1) * this->tileDataNum;
      uint32_t prev_remaining = this->my_core_data - prev_start;
      uint32_t prev_logical =
          (prev_remaining > this->tileDataNum) ? this->tileDataNum : prev_remaining;
      uint32_t prev_compute = (prev_logical + align_mask) & ~align_mask;
      if (prev_compute > this->tileDataNum)
        prev_compute = this->tileDataNum;

      uint32_t cur_start = i * this->tileDataNum;
      uint32_t cur_remaining = this->my_core_data - cur_start;
      uint32_t cur_logical =
          (cur_remaining > this->tileDataNum) ? this->tileDataNum : cur_remaining;
      uint32_t cur_compute = (cur_logical + align_mask) & ~align_mask;
      if (cur_compute > this->tileDataNum)
        cur_compute = this->tileDataNum;
      CopyIn(static_cast<int32_t>(i), cur_logical, cur_compute, cur_start);
      Compute(static_cast<int32_t>(i - 1), prev_compute, prev_logical,
              prev_start);
      CopyOut(static_cast<int32_t>(i - 1), prev_logical, prev_start);
    }
    {
      uint32_t last_idx = actual_tile_num - 1;
      uint32_t last_start = last_idx * this->tileDataNum;
      uint32_t last_remaining = this->my_core_data - last_start;
      uint32_t last_logical =
          (last_remaining > this->tileDataNum) ? this->tileDataNum : last_remaining;
      uint32_t last_compute = (last_logical + align_mask) & ~align_mask;
      if (last_compute > this->tileDataNum)
        last_compute = this->tileDataNum;
      Compute(static_cast<int32_t>(last_idx), last_compute, last_logical,
              last_start);
      CopyOut(static_cast<int32_t>(last_idx), last_logical, last_start);
    }
  }

private:
  __aicore__ inline void CopyLocal(LocalTensor<float> &dst,
                                   LocalTensor<float> &src,
                                   uint32_t calCount) {
    const uint32_t vw = 32 / sizeof(float);
    uint32_t mid = (calCount / vw) * vw;
    if (mid > 0) {
      DataCopy(dst[0], src[0], mid);
    }
    for (uint32_t i = mid; i < calCount; ++i) {
      dst.SetValue(i, src.GetValue(i));
    }
  }
  template <typename TDst, typename TSrc>
  __aicore__ inline void CopyLocal(LocalTensor<TDst> &dst,
                                   LocalTensor<TSrc> &src,
                                   uint32_t calCount) {
    Cast(dst, src, RoundMode::CAST_NONE, calCount);
  }
  __aicore__ inline void CopyIn(int32_t progress, uint32_t logicalNum,
                                uint32_t computeNum, uint32_t local_offset) {
    LocalTensor<DTYPE_INPUT> inputLocal =
        inQueueInput.AllocTensor<DTYPE_INPUT>();
    LocalTensor<DTYPE_OTHER> otherLocal =
        inQueueOther.AllocTensor<DTYPE_OTHER>();
    BroadCastCopy(inputLocal, local_offset, logicalNum, computeNum, 0,
                  this->input_Length, inputGm);
    BroadCastCopy(otherLocal, local_offset, logicalNum, computeNum, 1,
                  this->other_Length, otherGm);
    inQueueInput.EnQue(inputLocal);
    inQueueOther.EnQue(otherLocal);
  }

  template <typename T, typename T_GM>
  __aicore__ inline void
  BroadCastCopy(LocalTensor<T> &dst, uint32_t local_offset, uint32_t logicalNum,
                uint32_t computeNum, int inputNo, uint64_t gm_global_length,
                GlobalTensor<T_GM> &gm) {
    uint32_t vec_width = 32 / sizeof(T);
    bool is_broadcast = (gm_global_length != this->total_length);

    if (gm_global_length == 1ULL) {
      T tmp = gm.GetValue(0);
      if constexpr (std::is_same<T, bool>::value) {
        for (uint32_t j = 0; j < computeNum; ++j) {
          dst.SetValue(j, tmp);
        }
      } else {
        const uint32_t vec_width_loc = 32 / sizeof(T);
        uint32_t processed_loc = 0;
        if (computeNum > 0) {
          uint32_t head_need = (vec_width_loc - (0 % vec_width_loc)) % vec_width_loc;
          uint32_t head_cnt = (computeNum < head_need) ? computeNum : head_need;
          for (uint32_t j = 0; j < head_cnt; ++j) { dst.SetValue(processed_loc + j, tmp); }
          processed_loc += head_cnt;
        }
        uint32_t remain_loc = (computeNum > processed_loc) ? (computeNum - processed_loc) : 0;
        uint32_t mid_aligned = (remain_loc / vec_width_loc) * vec_width_loc;
        if (mid_aligned > 0) {
          for (uint32_t j = 0; j < vec_width_loc; ++j) { dst.SetValue(processed_loc + j, tmp); }
          uint32_t filled = vec_width_loc;
          while ((filled << 1) <= mid_aligned) {
            DataCopy(dst[processed_loc + filled], dst[processed_loc], filled);
            filled <<= 1;
          }
          if (filled < mid_aligned) {
            DataCopy(dst[processed_loc + filled], dst[processed_loc], mid_aligned - filled);
            filled = mid_aligned;
          }
          processed_loc += mid_aligned;
        }
        for (uint32_t j = processed_loc; j < computeNum; ++j) { dst.SetValue(j, tmp); }
      }
      return;
    }

    if (!is_broadcast) {
      uint32_t processed = 0;
      if (logicalNum > 0) {
        uint32_t head_need = (vec_width - (local_offset % vec_width)) % vec_width;
        uint32_t head_cnt = (logicalNum < head_need) ? logicalNum : head_need;
        for (uint32_t j = 0; j < head_cnt; ++j) {
          dst.SetValue(processed + j, gm.GetValue(local_offset + j));
        }
        processed += head_cnt;
        local_offset += head_cnt;
      }
      uint32_t remain = (logicalNum > processed) ? (logicalNum - processed) : 0;
      if (remain >= vec_width && !std::is_same<T, bool>::value) {
        uint32_t mid = (remain / vec_width) * vec_width;
        if (mid > 0) {
          DataCopy(dst[processed], gm[local_offset], mid);
          processed += mid;
          local_offset += mid;
        }
      }
      for (uint32_t j = processed; j < logicalNum; ++j) {
        dst.SetValue(j, gm.GetValue(local_offset + (j - processed)));
      }
      if (computeNum > logicalNum) {
        for (uint32_t j = logicalNum; j < computeNum; ++j) {
          dst.SetValue(j, static_cast<T>(0));
        }
      }
      return;
    }

    const int64_t dimOffset = maxDimNum - this->numshapes;
    const int64_t lastDim = dimOffset + this->numshapes - 1;
    const uint64_t outLast = static_cast<uint64_t>(this->shapefull[lastDim] > 0 ? this->shapefull[lastDim] : 1);
    const uint64_t inLast  = static_cast<uint64_t>(this->shape[inputNo][lastDim] > 0 ? this->shape[inputNo][lastDim] : 1);

    uint32_t processed = 0;
    while (processed < logicalNum) {
      uint64_t global_out_idx = this->core_start + static_cast<uint64_t>(local_offset) + processed;
      uint64_t input_base_idx64 = GetPos(global_out_idx, inputNo);
      if (input_base_idx64 > UINT32_MAX) {
        uint32_t remain2 = logicalNum - processed;
        for (uint32_t k = 0; k < remain2; ++k) {
          dst.SetValue(processed + k, static_cast<T>(0));
        }
        break;
      }
      uint32_t input_base_idx = static_cast<uint32_t>(input_base_idx64);
      uint64_t offset_in_last = (outLast > 0) ? (global_out_idx % outLast) : 0ULL;
      uint32_t n_to_boundary = static_cast<uint32_t>(outLast - offset_in_last);
      uint32_t seg = (logicalNum - processed < n_to_boundary) ? (logicalNum - processed) : n_to_boundary;

      if (inLast == outLast) {
        uint32_t aligned = 0;
        bool src_aligned = ((input_base_idx % vec_width) == 0);
        bool dst_aligned = ((processed % vec_width) == 0);
        if (src_aligned && dst_aligned) {
          aligned = (seg / vec_width) * vec_width;
          if (aligned > 0) {
            DataCopy(dst[processed], gm[input_base_idx], aligned);
          }
        }
        for (uint32_t k = aligned; k < seg; ++k) {
          dst.SetValue(processed + k, gm.GetValue(input_base_idx + k));
        }
      } else {
        T val = gm.GetValue(input_base_idx);
        if constexpr (std::is_same<T, bool>::value) {
          for (uint32_t k = 0; k < seg; ++k) { dst.SetValue(processed + k, val); }
        } else {
          uint32_t vec_width_loc = 32 / sizeof(T);
          uint32_t head_need = (vec_width_loc - (processed % vec_width_loc)) % vec_width_loc;
          uint32_t head_cnt = (seg < head_need) ? seg : head_need;
          for (uint32_t j = 0; j < head_cnt; ++j) { dst.SetValue(processed + j, val); }
          uint32_t done = head_cnt;
          uint32_t remain_loc = seg - done;
          uint32_t mid_aligned = (remain_loc / vec_width_loc) * vec_width_loc;
          if (mid_aligned > 0) {
            for (uint32_t j = 0; j < vec_width_loc; ++j) { dst.SetValue(processed + done + j, val); }
            uint32_t filled = vec_width_loc;
            while ((filled << 1) <= mid_aligned) {
              DataCopy(dst[processed + done + filled], dst[processed + done], filled);
              filled <<= 1;
            }
            if (filled < mid_aligned) {
              DataCopy(dst[processed + done + filled], dst[processed + done], mid_aligned - filled);
              filled = mid_aligned;
            }
            done += mid_aligned;
          }
          for (uint32_t j = done; j < seg; ++j) { dst.SetValue(processed + j, val); }
        }
      }
      processed += seg;
    }
    if (computeNum > logicalNum) {
      for (uint32_t j = logicalNum; j < computeNum; ++j) {
        dst.SetValue(j, static_cast<T>(0));
      }
    }
  }

  __aicore__ inline uint64_t GetPos(uint64_t linear_output_idx, int inputNo) {
    if (linear_output_idx >= this->total_length || this->numshapes <= 0) {
      return 0ULL;
    }
    const int64_t dimOffset = maxDimNum - this->numshapes;
    uint64_t remaining = linear_output_idx;
    uint64_t linear_input_idx = 0ULL;
    for (int32_t i = 0; i < this->numshapes; ++i) {
      int64_t out_dim = this->shapefull[dimOffset + i];
      int64_t in_dim = this->shape[inputNo][dimOffset + i];
      uint64_t stride = this->outputStrides[dimOffset + i];
      uint64_t coord = (out_dim > 1) ? (remaining / stride) : 0ULL;
      remaining -= coord * stride;
      if (in_dim == 1) { coord = 0ULL; }
      if (i == 0) {
        linear_input_idx = coord;
      } else {
        linear_input_idx = linear_input_idx * static_cast<uint64_t>(in_dim > 0 ? in_dim : 1) + coord;
      }
    }
    return linear_input_idx;
  }

  __aicore__ inline void Compute(int32_t progress, uint32_t computeNum,
                                 uint32_t logicalNum, uint32_t local_offset) {
    if (computeNum == 0 || logicalNum == 0)
      return;
    LocalTensor<DTYPE_INPUT> inputLocal = inQueueInput.DeQue<DTYPE_INPUT>();
    LocalTensor<DTYPE_OTHER> otherLocal = inQueueOther.DeQue<DTYPE_OTHER>();
    LocalTensor<DTYPE_OUT> outLocal = outQueueOut.AllocTensor<DTYPE_OUT>();

    uint32_t limit = logicalNum;
    if (limit > this->tileDataNum) limit = this->tileDataNum;

    if constexpr (std::is_same_v<DTYPE_OUT, bool>) {
      if (this->other_is_scalar && this->other_scalar_bool) {
        for (uint32_t i = 0; i < limit; ++i) { outLocal.SetValue(i, true); }
      } else {
        if constexpr (!std::is_same_v<DTYPE_INPUT, bool>) {
          LocalTensor<float> xF = tmpBuf1.Get<float>();
          Cast(xF, inputLocal, RoundMode::CAST_NONE, limit);
          if constexpr (std::is_same_v<DTYPE_OTHER, bool>) {
            for (uint32_t i = 0; i < limit; ++i) {
              bool xi = (xF.GetValue(i) != 0.0f);
              bool yi = this->other_is_scalar ? this->other_scalar_bool : otherLocal.GetValue(i);
              outLocal.SetValue(i, xi || yi);
            }
          } else {
            LocalTensor<float> yF = tmpBuf2.Get<float>();
            Cast(yF, otherLocal, RoundMode::CAST_NONE, limit);
            for (uint32_t i = 0; i < limit; ++i) {
              bool xi = (xF.GetValue(i) != 0.0f);
              bool yi = (yF.GetValue(i) != 0.0f);
              outLocal.SetValue(i, xi || yi);
            }
          }
        } else {
          if constexpr (std::is_same_v<DTYPE_OTHER, bool>) {
            for (uint32_t i = 0; i < limit; ++i) {
              bool xi = this->input_is_scalar ? this->input_scalar_bool : inputLocal.GetValue(i);
              bool yi = this->other_is_scalar ? this->other_scalar_bool : otherLocal.GetValue(i);
              outLocal.SetValue(i, xi || yi);
            }
          } else {
            LocalTensor<float> yF = tmpBuf2.Get<float>();
            Cast(yF, otherLocal, RoundMode::CAST_NONE, limit);
            for (uint32_t i = 0; i < limit; ++i) {
              bool xi = this->input_is_scalar ? this->input_scalar_bool : inputLocal.GetValue(i);
              bool yi = (yF.GetValue(i) != 0.0f);
              outLocal.SetValue(i, xi || yi);
            }
          }
        }
      }
    } else if constexpr (std::is_same_v<DTYPE_OUT, half> || std::is_same_v<DTYPE_OUT, bfloat16_t>) {
      LocalTensor<float> xF = tmpBuf1.Get<float>();
      LocalTensor<float> yF = tmpBuf2.Get<float>();
      Cast(xF, inputLocal, RoundMode::CAST_NONE, limit);
      Cast(yF, otherLocal, RoundMode::CAST_NONE, limit);
      Max(xF, xF, yF, limit);  // 核心修改：Min -> Max
      Cast(outLocal, xF, RoundMode::CAST_TRUNC, limit);
    } else if constexpr (std::is_same_v<DTYPE_OUT, float>) {
      Max(outLocal, inputLocal, otherLocal, limit);  // 核心修改：Min -> Max
    } else {
      if constexpr (std::is_same_v<DTYPE_INPUT, DTYPE_OUT> && std::is_same_v<DTYPE_OTHER, DTYPE_OUT> &&
                    (std::is_same_v<DTYPE_OUT, int16_t> || std::is_same_v<DTYPE_OUT, uint16_t> ||
                     std::is_same_v<DTYPE_OUT, int32_t> || std::is_same_v<DTYPE_OUT, uint32_t>)) {
        Max(outLocal, inputLocal, otherLocal, limit);  // 核心修改：Min -> Max
      } else {
        for (uint32_t i = 0; i < limit; ++i) {
          long long xv = (long long)static_cast<long long>(inputLocal.GetValue(i));
          long long yv = (long long)static_cast<long long>(otherLocal.GetValue(i));
          long long rv = (xv > yv ? xv : yv);  // 核心修改：< -> >
          outLocal.SetValue(i, static_cast<DTYPE_OUT>(rv));
        }
      }
    }

    outQueueOut.EnQue<DTYPE_OUT>(outLocal);
    inQueueInput.FreeTensor(inputLocal);
    inQueueOther.FreeTensor(otherLocal);
  }

  __aicore__ inline void CopyOut(int32_t progress, uint32_t logicalNum,
                                 uint32_t local_offset) {
    LocalTensor<DTYPE_OUT> outLocal = outQueueOut.DeQue<DTYPE_OUT>();
    if constexpr (std::is_same_v<DTYPE_OUT, bool>) {
      for (uint32_t j = 0; j < logicalNum; ++j) {
        outGm.SetValue(local_offset + j, outLocal.GetValue(j));
      }
      outQueueOut.FreeTensor(outLocal);
      return;
    }
    uint32_t vec_width = 32 / sizeof(DTYPE_OUT);
    uint32_t processed = 0;
    if (logicalNum > 0) {
      uint32_t head_need = (vec_width - (local_offset % vec_width)) % vec_width;
      uint32_t head_cnt = (logicalNum < head_need) ? logicalNum : head_need;
      for (uint32_t j = 0; j < head_cnt; ++j) {
        outGm.SetValue(local_offset + j, outLocal.GetValue(j));
      }
      processed += head_cnt;
      local_offset += head_cnt;
    }
    uint32_t remain = (logicalNum > processed) ? (logicalNum - processed) : 0;
    if (remain >= vec_width && !std::is_same<DTYPE_OUT, bool>::value) {
      uint32_t mid = (remain / vec_width) * vec_width;
      if (mid > 0) {
        auto out_view = outGm[local_offset];
        DataCopy(out_view, outLocal[processed], mid);
        processed += mid;
        local_offset += mid;
      }
    }
    for (uint32_t j = processed; j < logicalNum; ++j) {
      outGm.SetValue(local_offset + (j - processed), outLocal.GetValue(j));
    }
    outQueueOut.FreeTensor(outLocal);
  }

private:
  TPipe *pipe;
  TQue<QuePosition::VECIN, BUFFER_NUM> inQueueInput, inQueueOther;
  TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueOut;
  TBuf<QuePosition::VECCALC> tmpBuf1, tmpBuf2;
  GlobalTensor<DTYPE_INPUT> inputGm;
  GlobalTensor<DTYPE_OTHER> otherGm;
  GlobalTensor<DTYPE_OUT> outGm;

  uint32_t coreDataNum;
  uint64_t tileNum64;
  uint32_t tileDataNum;
  uint32_t tailDataNum;

  uint64_t input_Length;
  uint64_t other_Length;
  uint64_t total_length;
  uint64_t my_core_data;
  uint64_t core_start;

  int64_t numshapes;
  int64_t shape[inputVarNum][maxDimNum];
  int64_t shapefull[maxDimNum];
  uint64_t outputStrides[maxDimNum];
  uint64_t inputStrides[inputVarNum][maxDimNum];

  bool input_is_scalar;
  bool other_is_scalar;
  bool input_scalar_bool;
  bool other_scalar_bool;
};

extern "C" __global__ __aicore__ void fmax(GM_ADDR input, GM_ADDR other,
                                            GM_ADDR out, GM_ADDR workspace,
                                            GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
  TPipe pipe;
  KernelFmax_Broadcast<DTYPE_INPUT, DTYPE_OTHER, DTYPE_OUT> op;
  TileVar tilevar;
  tilevar.CoreDataNum = tiling_data.CoreDataNum;
  tilevar.finalTileNum = tiling_data.finalTileNum;
  tilevar.tileDataNum = tiling_data.tileDataNum;
  tilevar.TailDataNum = tiling_data.TailDataNum;
  tilevar.input_length = tiling_data.InputLength;
  tilevar.other_length = tiling_data.OtherLength;
  tilevar.numshapes = tiling_data.numshapes;
  for (int32_t i = 0; i < inputVarNum * maxDimNum; i++) {
    tilevar.ss[i] = tiling_data.shape[i];
  }
  for (int32_t i = 0; i < maxDimNum; i++) {
    tilevar.sf[i] = tiling_data.shapefull[i];
  }
  op.Init(input, other, out, &tilevar, &pipe);
  op.Process();
}
