# Copyright 2025 The pix2struct Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

def relaxed_correctness(target: str,
                            prediction: str,
                            max_relative_change: float = 0.05) -> bool:
        """Calculates relaxed correctness.

        The correctness tolerates certain error ratio defined by max_relative_change.
        See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
        “Following Methani et al. (2020), we use a relaxed accuracy measure for the
        numeric answers to allow a minor inaccuracy that may result from the automatic
        data extraction process. We consider an answer to be correct if it is within
        5% of the gold answer. For non-numeric answers, we still need an exact match
        to consider an answer to be correct.”

        Args:
            target: Target string.
            prediction: Predicted string.
            max_relative_change: Maximum relative change.

        Returns:
            Whether the prediction was correct given the specified tolerance.
        """

        def _to_float(text: str) -> Optional[float]:
            try:
                if text.endswith("%"):
                    # Convert percentages to floats.
                    return float(text.rstrip("%")) / 100.0
                else:
                    return float(text)
            except ValueError:
                return None

        prediction_float = _to_float(prediction)
        target_float = _to_float(target)
        if prediction_float is not None and target_float:
            relative_change = abs(prediction_float - target_float) / abs(target_float)
            return relative_change <= max_relative_change
        else:
            return prediction.lower() == target.lower()

def exact_correctness(target: str, prediction: str):
    return prediction.strip().lower() == target.strip().lower()