
// Licensed under the Apache License, Version 2.0 (the 'License');
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an 'AS IS' BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Copyright 2005-2016 Brian Roark and Google, Inc.
#include <ngram/ngram-list-prune.h>
#include <ngram/ngram-input.h>

namespace ngram {

void GetNGramListToPrune(
    const std::vector<string> &ngrams_to_prune,
    const fst::SymbolTable *syms,
    std::set<std::vector<fst::StdArc::Label>> *ngram_list) {
  if (syms == nullptr) {
    LOG(WARNING) << "empty symbol table, no means for compiling ngram list.";
    return;
  }
  if (ngrams_to_prune.empty()) {
    LOG(WARNING) << "vector of ngram strings empty, no list compiled.";
  }
  for (auto ngram_string : ngrams_to_prune) {
    if (!ngram_string.empty()) {
      std::vector<string> ngram_words;
      ReadTokenString(ngram_string, &ngram_words);
      if (!ngram_words.empty()) {
        std::vector<fst::StdArc::Label> ngram_labels(ngram_words.size());
        bool all_found = true;
        for (int i = 0; i < ngram_words.size(); ++i) {
          ngram_labels[i] = syms->Find(ngram_words[i]);
          if (ngram_labels[i] < 0) {
            all_found = false;
            LOG(WARNING) << "word not in symbol table, ngram not added: "
                         << ngram_string;
            break;
          }
        }
        if (all_found) {
          ngram_list->insert(ngram_labels);
        }
      }
    }
  }
}

}  // namespace ngram
