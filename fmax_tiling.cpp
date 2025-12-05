#include "fmax_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

constexpr int32_t inputVarNum = 2;
constexpr int32_t maxDimNum = 64;

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
  FmaxTilingData tiling;
  uint64_t ubSize;
  auto ascendcPlatform =
      platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
  uint32_t inputBytes =
      GetSizeByDataType(context->GetInputDesc(0)->GetDataType());
  uint32_t otherBytes =
      GetSizeByDataType(context->GetInputDesc(1)->GetDataType());
  uint32_t outputBytes =
      GetSizeByDataType(context->GetOutputDesc(0)->GetDataType());
  inputBytes = (inputBytes == 0) ? GetSizeByDataType(ge::DT_FLOAT) : inputBytes;
  otherBytes = (otherBytes == 0) ? GetSizeByDataType(ge::DT_FLOAT) : otherBytes;
  outputBytes =
      (outputBytes == 0) ? GetSizeByDataType(ge::DT_FLOAT) : outputBytes;

  uint32_t totalBytesPerElement = 3 * inputBytes + 3 * otherBytes +
                                  3 * outputBytes +
                                  2 * 4U;
  
  uint32_t tileDataNumMax = ubSize / totalBytesPerElement;

  auto gcd_u32 = [](uint32_t a, uint32_t b) -> uint32_t {
    if (a == 0)
      return b;
    if (b == 0)
      return a;
    while (b) {
      uint32_t t = a % b;
      a = b;
      b = t;
    }
    return a;
  };
  auto lcm_u32 = [&](uint32_t a, uint32_t b) -> uint32_t {
    if (a == 0 || b == 0)
      return a + b;
    uint64_t g = gcd_u32(a, b);
    uint64_t l = (uint64_t)a / g * (uint64_t)b;
    return (l > UINT32_MAX) ? UINT32_MAX : (uint32_t)l;
  };
  auto elemsAlignBytes = [&](uint32_t bytes, uint32_t alignBytes) -> uint32_t {
    if (bytes == 0)
      return 1;
    uint32_t g = gcd_u32(bytes, alignBytes);
    return alignBytes / g;
  };
  auto elemsAlign32 = [&](uint32_t bytes) -> uint32_t {
    if (bytes == 0)
      return 1U;
    return 32U / gcd_u32(bytes, 32U);
  };
  uint32_t align512 = lcm_u32(lcm_u32(elemsAlignBytes(inputBytes, 512),
                                      elemsAlignBytes(otherBytes, 512)),
                              elemsAlignBytes(outputBytes, 512));
  uint32_t align32 =
      lcm_u32(lcm_u32(elemsAlign32(inputBytes), elemsAlign32(otherBytes)),
              elemsAlign32(outputBytes));
  if (align32 < 1)
    align32 = 1;

  auto minElemsForBytes = [](uint32_t bytes) -> uint32_t {
    if (bytes == 0)
      return 1;
    uint32_t need = (16384U + bytes - 1U) / bytes;
    if (need < (32U / (bytes == 0 ? 1 : bytes)))
      need = (32U / (bytes == 0 ? 1 : bytes));
    return need == 0 ? 1U : need;
  };
  uint32_t m1 = minElemsForBytes(inputBytes);
  uint32_t m2 = minElemsForBytes(otherBytes);
  uint32_t m3 = minElemsForBytes(outputBytes);
  uint32_t min16KB = (m1 > m2 ? m1 : m2);
  if (m3 > min16KB)
    min16KB = m3;

  uint32_t tileDataNum512 = (tileDataNumMax / align512) * align512;
  uint32_t tileDataNum32 = (tileDataNumMax / align32) * align32;
  uint32_t tileDataNum = 0;
  if (tileDataNum512 >= min16KB) {
    tileDataNum = tileDataNum512;
  } else if (tileDataNum32 >= min16KB) {
    tileDataNum = tileDataNum32;
  } else {
    tileDataNum = std::max(tileDataNum512, tileDataNum32);
    if (tileDataNum == 0)
      tileDataNum = align32;
  }
  if (tileDataNum > 0 && tileDataNum < 4096U) {
    tileDataNum = (tileDataNum / align32) * align32;
    if (tileDataNum == 0) tileDataNum = align32;
  }

  int64_t input_dims = context->GetInputShape(0)->GetStorageShape().GetDimNum();
  int64_t other_dims = context->GetInputShape(1)->GetStorageShape().GetDimNum();
  int64_t numshapes = (input_dims > other_dims) ? input_dims : other_dims;
  

  int64_t shape[maxDimNum * inputVarNum], shapefull[maxDimNum];
  memset(shape, 0, sizeof(shape));
  memset(shapefull, 0, sizeof(shapefull));
  int64_t *input_shape_arr = &shape[0 * maxDimNum];
  for (int i = 0; i < maxDimNum - input_dims; i++)
    input_shape_arr[i] = 1;
  for (int i = 0; i < input_dims; i++) {
    input_shape_arr[maxDimNum - input_dims + i] =
        context->GetInputShape(0)->GetStorageShape().GetDim(i);
  }
  int64_t *other_shape_arr = &shape[1 * maxDimNum];
  for (int i = 0; i < maxDimNum - other_dims; i++)
    other_shape_arr[i] = 1;
  for (int i = 0; i < other_dims; i++) {
    other_shape_arr[maxDimNum - other_dims + i] =
        context->GetInputShape(1)->GetStorageShape().GetDim(i);
  }

  uint64_t total_output_elements = 1ULL;
  const int64_t dimOffset = maxDimNum - numshapes;
  bool overflow = false;
  bool has_zero_dim = false;
  for (int k = 0; k < numshapes; ++k) {
    int64_t a = input_shape_arr[dimOffset + k];
    int64_t b = other_shape_arr[dimOffset + k];
    
    int64_t c = (a == 1) ? b : (b == 1 ? a : a);
    shapefull[dimOffset + k] = c;
    if (c == 0) {
      total_output_elements = 0ULL;
      has_zero_dim = true;
      break;
    }
    if (total_output_elements > UINT64_MAX / (uint64_t)c) {
      overflow = true;
      break;
    }
    total_output_elements *= (uint64_t)c;
  }
  if (total_output_elements > 0 && tileDataNum > total_output_elements) {
    uint32_t adj = static_cast<uint32_t>(total_output_elements > UINT32_MAX ? UINT32_MAX : total_output_elements);
    if (adj > 0) {
      tileDataNum = (adj / align32) * align32;
      if (tileDataNum == 0) tileDataNum = align32;
    }
  }
  
  for (int i = 0; i < dimOffset; i++)
    shapefull[i] = 1;

  uint64_t available_cores = ascendcPlatform.GetCoreNum();
  if (available_cores == 0)
    available_cores = 25ULL;
  uint64_t block_dim_64 = 1ULL;
  if (total_output_elements > 0) {
    block_dim_64 = available_cores;
    uint64_t max_core_data =
        (total_output_elements + block_dim_64 - 1ULL) / block_dim_64;
    if (max_core_data < tileDataNum) {
      block_dim_64 = (total_output_elements + tileDataNum - 1ULL) / tileDataNum;
    }
    if (total_output_elements <= 8192ULL) {
      block_dim_64 = 1ULL;
    }
  
    uint32_t CoreDataNum = static_cast<uint32_t>(
        max_core_data > UINT32_MAX ? UINT32_MAX : max_core_data);
    uint32_t TileNum = (CoreDataNum + tileDataNum - 1U) / tileDataNum;
    if (TileNum == 0 && CoreDataNum > 0)
      TileNum = 1;
    uint32_t finalTileNum = TileNum;
    uint32_t TailDataNum = CoreDataNum % tileDataNum;
    if (TailDataNum == 0 && CoreDataNum > 0)
      TailDataNum = tileDataNum;
    if (CoreDataNum == 0) {
      finalTileNum = 0;
      TailDataNum = 0;
    }
    tiling.set_CoreDataNum(CoreDataNum);
    tiling.set_finalTileNum(finalTileNum);
    tiling.set_tileDataNum(tileDataNum);
    tiling.set_TailDataNum(TailDataNum);
    context->SetBlockDim(static_cast<int32_t>(block_dim_64));
  } else {
    context->SetBlockDim(1);
    tiling.set_CoreDataNum(0);
    tiling.set_finalTileNum(0);
    tiling.set_TailDataNum(0);
  }

  uint64_t input_length =
      context->GetInputShape(0)->GetStorageShape().GetShapeSize();
  uint64_t other_length =
      context->GetInputShape(1)->GetStorageShape().GetShapeSize();
  if (input_length > UINT32_MAX || other_length > UINT32_MAX) {
    return ge::GRAPH_FAILED;
  }
  tiling.set_InputLength(input_length);
  tiling.set_OtherLength(other_length);
  tiling.set_numshapes(numshapes);
  tiling.set_shape(shape);
  tiling.set_shapefull(shapefull);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                      context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  size_t *currentWorkspace = context->GetWorkspaceSizes(1);
  currentWorkspace[0] = 0;
  return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context) {
  const gert::Shape *input_shape = context->GetInputShape(0);
  const gert::Shape *other_shape = context->GetInputShape(1);
  gert::Shape *out_shape = context->GetOutputShape(0);
  int64_t input_dims = input_shape->GetDimNum();
  int64_t other_dims = other_shape->GetDimNum();
  int64_t max_dims = (input_dims > other_dims) ? input_dims : other_dims;
 
  std::vector<int64_t> result_dims(max_dims, 1);
  for (int64_t i = 0; i < max_dims; i++) {
    int64_t a = (i < input_dims) ? input_shape->GetDim(input_dims - 1 - i) : 1;
    int64_t b = (i < other_dims) ? other_shape->GetDim(other_dims - 1 - i) : 1;
   
    int64_t c;
    if (a == 1)
      c = b;
    else if (b == 1)
      c = a;
    else if (a == b)
      c = a;
    
    result_dims[max_dims - 1 - i] = c;
  }
  out_shape->SetDimNum(max_dims);
  for (int64_t i = 0; i < max_dims; i++)
    out_shape->SetDim(i, result_dims[i]);
  return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context) {
  const auto inputDataType = context->GetInputDataType(0);
  context->SetOutputDataType(0, inputDataType);
  return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class Fmax : public OpDef {
public:
  explicit Fmax(const char *name) : OpDef(name) {
    this->Input("input")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BOOL, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32,
                   ge::DT_INT8, ge::DT_INT64, ge::DT_INT16, ge::DT_UINT8,ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND,ge::FORMAT_ND});
    this->Input("other")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BOOL, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32,
                   ge::DT_INT8, ge::DT_INT64, ge::DT_INT16, ge::DT_UINT8,ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("out")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BOOL, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32,
                   ge::DT_INT8, ge::DT_INT64, ge::DT_INT16, ge::DT_UINT8,ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND,ge::FORMAT_ND});

    this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
    this->AICore().SetTiling(optiling::TilingFunc);
    this->AICore().AddConfig("ascend910b");
  }
};

OP_ADD(Fmax);
} // namespace ops
