#include <cassert>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <set>
#include <vector>
#include <chrono> 
#include <algorithm>
#include <bits/stdc++.h>

#include "fptree.hpp"


using namespace std::chrono;
using namespace std;

int label;
long long int max_item;

bool compare(const pair<vector<string>, int>& a, const pair<vector<string>, int>& b) {
    if (a.first.size() != b.first.size()) {
        return a.first.size() > b.first.size();
    }
    return a.second > b.second; 
}

bool numericComparator(const std::string &a, const std::string &b) {
    return std::stoll(a) < std::stoll(b);
}

void f(Transaction& transaction, pair<Transaction, int64_t>& itemset, unordered_map<Item,Transaction>& decoder){
    int i = 0;
    int j = 0;

    vector<string> new_transaction = {};
    while(i<transaction.size() && j<itemset.first.size()){
        if (transaction[i] == itemset.first[j]){
            i += 1;
            j += 1;
        }
        else{
            try{
                if (stoi(transaction[i])>stoi(itemset.first[j])){
                    break;
                }
            }
            catch(const std::invalid_argument& e){
                break;
            }
            new_transaction.push_back(transaction[i]);
            i += 1;
        }
    }

    if (j==itemset.first.size()){
        while(i<transaction.size()){
            new_transaction.push_back(transaction[i]);
            i += 1;
        }
        if (itemset.second>0){
            itemset.second = label;
            decoder[to_string(-1*label)] = itemset.first;
            new_transaction.push_back(to_string(-1*label));
            label -= 1;
        }
        else{
            new_transaction.push_back(to_string(-1*itemset.second));
        }
        sort(new_transaction.begin(),new_transaction.end(),numericComparator);
        transaction = new_transaction;
    }

}

void g(unordered_map<Item,Transaction>& decoder1, unordered_map<Item,Transaction>& decoder2){
    for (auto i:decoder2){
        Item key = i.first;
        Transaction curr = {};
        for(auto j:i.second){
            if(stoi(j) > max_item){
                for(auto k:decoder1[j]){
                    curr.push_back(k);
                }
            }  
            else{
                curr.push_back(j);
            }
        }
        decoder2[key] = curr;
    }
}

unordered_map<Item,Transaction> Compress(vector<Transaction>& transactions, uint64_t threshold, unordered_map<Item,Transaction>& pre_decoder){
    const FPTree fptree{ transactions, threshold };
    const std::set<Pattern> patterns = fptree_growth( fptree );

    vector<pair<Transaction,int64_t>> freq_items = {};
    for (auto i:patterns){
        if (i.first.size()>1) {
            vector<string> trans = {};
            for(auto j:i.first){
                trans.push_back(j);
            }
            sort(trans.begin(), trans.end(), numericComparator);
            freq_items.push_back({trans,i.second});
        }
    }

    sort(freq_items.begin(), freq_items.end(), compare);

    std::string freqItemsFileName = "freq_items_1500.txt";
    std::ofstream freqItemsFile(freqItemsFileName,std::ios::app);
    if (!freqItemsFile.is_open()) {
        std::cout << "Failed to open the output file." << std::endl;
    }

    for (const auto& i : freq_items) {
        const std::vector<std::string>& strings = i.first;
        int intValue = i.second;

        for (const std::string& str : strings) {
            freqItemsFile << str << " ";
        }

        freqItemsFile << "--- " << intValue << "\n";
    }
    freqItemsFile << "--------------------------------------------------------- "  << "\n";
    freqItemsFile.close();

    unordered_map<Item,Transaction> decoder;
    for (auto& transaction:transactions){
        for(auto& itemset:freq_items){
            if (transaction.size()<0){
                break;
            }
            f(transaction,itemset,decoder);
        }
        for (auto item:transaction){
            if (pre_decoder.find(item) != pre_decoder.end()){
                decoder[item] = pre_decoder[item];
            }
        }
    }

    return decoder;
}

void compress(string path_to_dataset, string path_to_output){

    std::ifstream inputFile(path_to_dataset); 
    if (!inputFile.is_open()) {
        std::cerr << "Failed to open the file." << std::endl;
    }

    std::vector<std::vector<std::string>> transactions;
    std::string line;
    while (std::getline(inputFile, line)) {
        std::vector<std::string> tokens;
        std::istringstream tokenStream(line);
        std::string token;
        
        while (tokenStream >> token) {
            tokens.push_back(token);
        }

        sort(tokens.begin(),tokens.end(),numericComparator);
        max_item = max(max_item,stoll(tokens[tokens.size() - 1]));
        transactions.push_back(tokens);
    }
    inputFile.close();


    label = -1*(max_item+1);

    uint64_t threshold = 10000;
    unordered_map<Item,Transaction> pre_decoder ={};
    unordered_map<Item,Transaction> decoder = Compress(transactions, threshold, pre_decoder);

    std::string decodedFileName = "decoded_1500.txt";
    std::ofstream decodedFile(decodedFileName);

    for (auto& i : transactions) {
        for (auto& j : i) {
            decodedFile << j << " ";
        }

        decodedFile << "\n";
    }

    decodedFile.close();

    std::string decoderFileName = "decoder_1500.txt";
    std::ofstream decoderFile(decoderFileName);

    for (auto& i : decoder) {
        decoderFile << i.first << "\n";
        for (auto& j : i.second) {
            decoderFile << j << " ";
        }
        decoderFile <<  "\n";
    }

    decoderFile.close();

    cout << "Done one recurrsion" << endl; 

    threshold = 9500;
    for(int i=0;i<48;i++){
        pre_decoder = decoder;
        decoder = Compress(transactions, threshold, pre_decoder);
        g(pre_decoder,decoder); 
        threshold -= 200;
        cout << "Done " << i+2 << "th  recurrsion" << endl;
    }

    std::string decodedFileName2 = "decoded_1500_2.txt";
    std::ofstream decodedFile2(decodedFileName2);

    for (auto& i : transactions) {
        for (auto& j : i) {
            decodedFile2 << j << " ";
        }

        decodedFile2 << "\n";
    }

    decodedFile2.close();

    std::string decoderFileName2 = "decoder_1500_2.txt";
    std::ofstream decoderFile2(decoderFileName2);

    for (auto& i : decoder) {
        decoderFile2 << i.first << "\n";
        for (auto& j : i.second) {
            decoderFile2 << j << " ";
        }
        decoderFile2 <<  "\n";
    }

    decoderFile2.close();

}

void decompress(string path_to_compressed_dataset, string path_to_reconstructed_dataset)
{
    ifstream compressedFile(path_to_compressed_dataset);
    if (!compressedFile.is_open())
    {
        cerr << "Failed to open the compressed file." << endl;
        return;
    }

    ofstream reconstructedFile(path_to_reconstructed_dataset);
    if (!reconstructedFile.is_open())
    {
        cerr << "Failed to open the output file." << endl;
        return;
    }

    // Read the decoder mapping from the compressed dataset
    map<string, string> decoderMap;
    string line;
    string key;
    bool isDecoderSection = true;

    while (getline(compressedFile, line))
    {
        // Check if we've reached the compressed data section
        if (line == "$$$")
        {
            isDecoderSection = false;
            continue;
        }

        if (isDecoderSection)
        {
            if (line[0] == 'L')
            {
                key = line;
            }
            else
            {
                decoderMap[key] = line;
            }
        }
        else
        {
            // Process the compressed dataset
            istringstream tokenStream(line);
            string token;
            while (tokenStream >> token)
            {
                if (decoderMap.find(token) != decoderMap.end())
                {
                    reconstructedFile << decoderMap[token] << " ";
                }
                else
                {
                    reconstructedFile << token << " ";
                }
            }
            reconstructedFile << "\n";
        }
    }

    compressedFile.close();
    reconstructedFile.close();
}


int main(int argc, const char *argv[])
{
    auto start = high_resolution_clock::now();
    if (argv[1] == "compress")
    {
        cout << "compress evolked";
        compress(argv[2], argv[3]);
    }
    else
    {
        decompress(argv[2], argv[3]);
    }

    auto end = high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(end - start);
    std::cout << "Execution Time: " << duration.count() << " ms\n";

    return EXIT_SUCCESS;
}
