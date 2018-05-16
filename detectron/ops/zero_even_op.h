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

#ifndef ZERO_EVEN_OP_H_
#define ZERO_EVEN_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

/**
 * ZeroEven operator. Zeros elements at even indices of an 1D array.
 * Elements at odd indices are preserved.
 *
 * This toy operator is an example of a custom operator and may be a useful
 * reference for adding new custom operators to the Detectron codebase.
 */
template <typename T, class Context>
class ZeroEvenOp final : public Operator<Context> {
 public:
  // Introduce Operator<Context> helper members.
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  ZeroEvenOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}

  bool RunOnDevice() override;
};

} // namespace caffe2

#endif // ZERO_EVEN_OP_H_
