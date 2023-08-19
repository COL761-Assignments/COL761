#include <cassert>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <set>
#include <vector>
#include <chrono> 
#include <algorithm>

#include "fptree.hpp"


using namespace std::chrono;
using namespace std;

int label = -1;

void g(vector<string>& transaction, pair<vector<string>,int64_t>& itemset){
    int i = 0;
    int j = 0;

    

    vector<string> new_transaction = {};
    while(i<transaction.size() && j<itemset.first.size()){
        if (transaction[i] == itemset.first[j]){
            i += 1;
            j += 1;
        }
        // else if (stoi(transaction[i])>stoi(itemset.first[j])){
        //     break;
        // }
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
            new_transaction.push_back("L"+to_string(-1*label));
            label -= 1;
        }
        else{
            new_transaction.push_back("L"+to_string(-1*itemset.second));
        }
        transaction = new_transaction;
    }

}

bool compare(const pair<vector<string>, int>& a, const pair<vector<string>, int>& b) {
    if (a.first.size() != b.first.size()) {
        return a.first.size() > b.first.size();
    }
    return a.second > b.second; 
}

bool numericComparator(const std::string &a, const std::string &b) {
    return std::stoi(a) < std::stoi(b);
}


void test_1()
{   
    auto start = high_resolution_clock::now();

    std::ifstream inputFile("/home/rutvik/Desktop/sem7/COL761/A1/D_medium.dat"); 
    // std::ifstream inputFile("/home/rutvik/Desktop/sem7/COL761/A1/test.dat"); 
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
        transactions.push_back(tokens);
    }
    inputFile.close();

    const uint64_t minimum_support_threshold = 5000;

    const FPTree fptree{ transactions, minimum_support_threshold };

    const std::set<Pattern> patterns = fptree_growth( fptree );


    auto end = high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(end - start);
    std::cout << "Execution Time: " << duration.count() << " ms\n";
    cout << "finding frequent itemsets completed " << patterns.size() << endl;

    vector<pair<vector<string>,int64_t>> freq_items = {};

    for (auto i:patterns){
        if (i.first.size()>1) { //i.first.size()>12 || (i.first.size()>1 && i.first.size()<5)){
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
    std::ofstream freqItemsFile(freqItemsFileName);
    if (!freqItemsFile.is_open()) {
        std::cout << "Failed to open the output file." << std::endl;
    }

    // Write each pair to the file
    for (const auto& i : freq_items) {
        const std::vector<std::string>& strings = i.first;
        int intValue = i.second;

        for (const std::string& str : strings) {
            freqItemsFile << str << " ";
        }

        freqItemsFile << "--- " << intValue << "\n";
    }

    // Close the output file
    freqItemsFile.close();


    for (auto& transaction:transactions){
        for(auto& itemset:freq_items){
            if (transaction.size()<0){
                break;
            }
            g(transaction,itemset);
        }
    }

    std::string decodedFileName = "decoded_1500.txt";
    std::ofstream decodedFile(decodedFileName);
    if (!decodedFile.is_open()) {
        std::cout << "Failed to open the output file." << std::endl;
    }

    for (auto& i : transactions) {
        for (auto& j : i) {
            decodedFile << j << " ";
        }

        decodedFile << "\n";
    }

    decodedFile.close();

    std::string decoderFileName = "decoder_1500.txt";
    std::ofstream decoderFile(decoderFileName);
    if (!decoderFile.is_open()) {
        std::cout << "Failed to open the output file." << std::endl;
    }

    for (auto& i : freq_items) {
        if (i.second < 0){
            decoderFile << "L"+to_string(-1*i.second) << "\n";
            for (auto& j : i.first) {
                decoderFile << j << " ";
            }
            decoderFile <<  "\n";
        }
    }

    decoderFile.close();

}



int main(int argc, const char *argv[])
{
    auto start = high_resolution_clock::now();

    test_1();

    auto end = high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(end - start);
    std::cout << "Execution Time: " << duration.count() << " ms\n";

    return EXIT_SUCCESS;
}
