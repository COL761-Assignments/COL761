#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <algorithm>
#include <bits/stdc++.h>

using namespace std;

using Transaction = vector<string>;
using Dataset = vector<Transaction>;

bool numericComparator(const std::string &a, const std::string &b) {
    return std::stoi(a) < std::stoi(b);
}


bool areEqualDatasets(const Dataset &old_Data, const Dataset &new_Data, const unordered_map<string, Transaction> &decoderMapping) {
    if (old_Data.size() != new_Data.size()) {
        return false;
    }

    uint64_t new_Data_size = 0;

    vector<vector<string>> new_decoded_Data = {};
    for (auto transaction:new_Data){
        new_Data_size  += transaction.size();
        vector<string> curr_transaction = {};
        for (auto item:transaction){
            if (decoderMapping.find(item) != decoderMapping.end()) {
                for (auto k: decoderMapping[item]){
                    curr_transaction.push_back(k);
                }
            }
            else{
                curr_transaction.push_back(item);
            }
        }
        new_decoded_Data.push_back(curr_transaction);
    }

    sort(new_decoded_Data.begin(), new_decoded_Data.end(), numericComparator);

    for (size_t i = 0; i < old_Data.size(); ++i) {
        if (old_Data[i] != new_decoded_Data[i]) {
            return false;
        }
    }

    // All checks passed, vectors are equal
    return true;

}

int main() {
    unordered_map<string, Transaction> decoderMapping = {
        {"L1", {"A","B", "C", "D"}},
        {"L2", {"E", "G"}}
    };
    Dataset originalDataset = {{"A", "B", "C", "D", "E"},
                                {"A", "B", "C", "D", "F"},
                                {"A", "B", "C", "D", "E", "G"},
                                {"A", "B", "C", "D", "E", "F", "G"}};

    Dataset newDataset = {{ "E","L1"},
                          {"F","L1"},
                          {"L1", "L2"},
                          {"G", "L1","L2"}};
    
    if (areEqualDatasets(originalDataset, newDataset, decoderMapping)){
        cout << "Equal" << endl;
    }
    else{
        cout << "Not Equal" << endl;
    }

    return 0;
}
