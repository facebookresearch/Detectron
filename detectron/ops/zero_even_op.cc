/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "zero_even_op.h"

namespace caffe2 {

template <>
bool ZeroEvenOp<float, CPUContext>::RunOnDevice() {
  // Retrieve the input tensor.
  const auto& X = Input(0);
  CAFFE_ENFORCE(X.dim() == 1);

  // Initialize the output tensor to a copy of the input tensor.
  auto* Y = Output(0);
  Y->CopyFrom(X);

  // Set output elements at even indices to zero.
  auto* Y_data = Y->mutable_data<float>();
  for (auto i = 0; i < Y->numel(); i += 2) {
    Y_data[i] = 0.0f;
  }

  return true;
}

REGISTER_CPU_OPERATOR(ZeroEven, ZeroEvenOp<float, CPUContext>);

OPERATOR_SCHEMA(ZeroEven)
    .NumInputs(1)
    .NumOutputs(1)
    .Input(
        0,
        "X",
        "1D input tensor")
    .Output(
        0,
        "Y",
        "1D output tensor");

} // namespace caffe2
