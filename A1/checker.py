def checker(decodedName, decoderName, TransactionName):
    decoder = {}
    decoder_length = 0

    decoderfile = open(decoderName, 'r')
    while True:
        key = decoderfile.readline()
        if not key: break
        value = decoderfile.readline()
        decoder[key.strip()] = [int(i) for i in value.strip().split(' ')]
        decoder_length += (1+len(decoder[key.strip()]))
    decoderfile.close()

    decoded = []
    decoded_length = 0
    decodedfile = open(decodedName, 'r')
    while True:
        line = decodedfile.readline()
        if not line: break
        decoded.append(line.strip().split(' '))
        decoded_length += len(decoded[-1])
    decodedfile.close()

    new_decoded = []
    for i in decoded:
        curr = []
        for j in i:
            try:
                curr = curr + decoder[j]
            except:
                curr.append(int(j))
        curr.sort()
        new_decoded.append(curr)
    
    # new_decoded.sort()
    
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

    # print(new_decoded)
    # print(' ')
    # print(original)

    if new_decoded == original:
        print("Equal")
        print("original Length ", original_length )
        print("decoder Length ", decoder_length )
        print("decoded Length ", decoded_length )
        print("Total Length ", decoder_length+decoded_length )  
        print("compression Percentage ",((original_length-decoded_length-decoder_length)*100)/original_length)
    else:
        print("Not Equal")



decodedName = '/home/rutvik/Downloads/FP-growth-master/decoded_1500.txt'
decoderName = '/home/rutvik/Downloads/FP-growth-master/decoder_1500.txt'
TransactionName = "/home/rutvik/Desktop/sem7/COL761/A1/D_medium.dat"

checker(decodedName, decoderName, TransactionName)

# A =[[1, 2, 3, 4, 5], [1, 2, 3, 4, 6], [1, 2, 3, 4, 5, 7], [1, 2, 3, 4, 5, 6, 7]]
# A.sort()
# print(A)