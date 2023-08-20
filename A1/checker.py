def checker(decodedName, outputName, TransactionName):

    
    original = []
    original_length = 0
    originalfile = open(TransactionName, 'r')
    while True:
        line = originalfile.readline()
        if not line: break
        curr = [int(i) for i in line.strip().split(' ')]
        curr.sort()
        original.append(curr)
        original_length += len(original[-1])
    originalfile.close()

    regenerated = []
    regenerated_length = 0
    regeneratedfile = open(decodedName, 'r')
    while True:
        line = regeneratedfile.readline()
        if not line: break
        curr = [int(i) for i in line.strip().split(' ')]
        curr.sort()
        regenerated.append(curr)
        regenerated_length += len(regenerated[-1])
    regeneratedfile.close()

    output_length = 0
    outputfile = open(outputName, 'r')
    while True:

        line = outputfile.readline()
        if not line: break
        curr = [i for i in line.strip().split(' ')]
        output_length += len(curr)
    outputfile.close()

    # print(new_decoded)
    # print(' ')
    # print(original)

    if regenerated == original:
        print("Equal")
        print("original Length ", original_length )
        print("Total Length ", output_length )  
        print("compression Percentage ",((original_length-output_length)*100)/original_length)
    else:
        print("Not Equal")



outputName = 'output.dat'
decodedName = 'original.dat'
TransactionName = "D_small.dat"

checker(decodedName, outputName, TransactionName)

# A =[[1, 2, 3, 4, 5], [1, 2, 3, 4, 6], [1, 2, 3, 4, 5, 7], [1, 2, 3, 4, 5, 6, 7]]
# A.sort()
# print(A)
