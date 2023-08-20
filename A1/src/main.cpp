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

void g(vector<string> &transaction, pair<vector<string>, int64_t> &itemset)
{
    int i = 0;
    int j = 0;

    vector<string> new_transaction = {};
    while (i < transaction.size() && j < itemset.first.size())
    {
        if (transaction[i] == itemset.first[j])
        {
            i += 1;
            j += 1;
        }
        // else if (stoi(transaction[i])>stoi(itemset.first[j])){
        //     break;
        // }
        else
        {
            try
            {
                if (stoi(transaction[i]) > stoi(itemset.first[j]))
                {
                    break;
                }
            }
            catch (const std::invalid_argument &e)
            {
                break;
            }
            new_transaction.push_back(transaction[i]);
            i += 1;
        }
    }

    if (j == itemset.first.size())
    {
        while (i < transaction.size())
        {
            new_transaction.push_back(transaction[i]);
            i += 1;
        }
        if (itemset.second > 0)
        {
            itemset.second = label;
            new_transaction.push_back("L" + to_string(-1 * label));
            label -= 1;
        }
        else
        {
            new_transaction.push_back("L" + to_string(-1 * itemset.second));
        }
        transaction = new_transaction;
    }
}

bool compare(const pair<vector<string>, int> &a, const pair<vector<string>, int> &b)
{
    if (a.first.size() != b.first.size())
    {
        return a.first.size() > b.first.size();
    }
    return a.second > b.second;
}

bool numericComparator(const std::string &a, const std::string &b)
{
    return std::stoi(a) < std::stoi(b);
}

void compress(string path_to_dataset, string path_to_output)
{
    auto start = high_resolution_clock::now();

    std::ifstream inputFile(path_to_dataset);
    if (!inputFile.is_open())
    {
        std::cerr << "Failed to open the file." << std::endl;
    }

    std::vector<std::vector<std::string>> transactions;

    std::string line;
    while (std::getline(inputFile, line))
    {
        std::vector<std::string> tokens;
        std::istringstream tokenStream(line);
        std::string token;

        while (tokenStream >> token)
        {
            tokens.push_back(token);
        }

        sort(tokens.begin(), tokens.end(), numericComparator);
        transactions.push_back(tokens);
    }
    inputFile.close();

    const uint64_t minimum_support_threshold = 2000;

    const FPTree fptree{transactions, minimum_support_threshold};

    const std::set<Pattern> patterns = fptree_growth(fptree);

    auto end = high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(end - start);
    std::cout << "Execution Time: " << duration.count() << " ms\n";
    cout << "finding frequent itemsets completed " << patterns.size() << endl;

    vector<pair<vector<string>, int64_t>> freq_items = {}; // This is unnecessary as sets are already sorted and can be accessed with a for loop.

    for (auto i : patterns)
    {
        if (i.first.size() > 1)
        { // i.first.size()>12 || (i.first.size()>1 && i.first.size()<5)){
            vector<string> trans = {};
            for (auto j : i.first)
            {
                trans.push_back(j);
            }
            sort(trans.begin(), trans.end(), numericComparator);
            freq_items.push_back({trans, i.second});
        }
    }

    sort(freq_items.begin(), freq_items.end(), compare);

    std::string freqItemsFileName = "freq_items_1500.txt";
    std::ofstream freqItemsFile(freqItemsFileName);
    if (!freqItemsFile.is_open())
    {
        std::cout << "Failed to open the output file." << std::endl;
    }

    // Write each pair to the file
    for (const auto &i : freq_items)
    {
        const std::vector<std::string> &strings = i.first;
        int intValue = i.second;

        for (const std::string &str : strings)
        {
            freqItemsFile << str << " ";
        }

        freqItemsFile << "--- " << intValue << "\n";
    }

    // Close the output file
    freqItemsFile.close();

    for (auto &transaction : transactions)
    {
        for (auto &itemset : freq_items)
        {
            if (transaction.size() < 0)
            { // is this necessary to check?
                break;
            }
            g(transaction, itemset);
        }
    }

    std::string decodedFileName = path_to_output;
    std::ofstream decodedFile(decodedFileName);
    if (!decodedFile.is_open())
    {
        std::cout << "Failed to open the output file." << std::endl;
    }

    for (auto &i : freq_items)
    {
        if (i.second < 0)
        {
            decodedFile << "L" + to_string(-1 * i.second) << "\n";
            for (auto &j : i.first)
            {
                decodedFile << j << " ";
            }
            decodedFile << "\n";
        }
    }
    decodedFile << "$$$"
                << "\n";

    for (auto &i : transactions)
    {
        for (auto &j : i)
        {
            decodedFile << j << " ";
        }

        decodedFile << "\n";
    }
    decodedFile.close();
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
