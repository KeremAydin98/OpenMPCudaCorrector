import os
from copy import deepcopy

def generateJSONL(prompData, completionData, outputFileName):
    FILE_PATH="jsonlFiles\\"
    outputFilePath = os.path.join(FILE_PATH, outputFileName)
    
    with open(outputFilePath, "w") as file:

        for i in range(len(prompData)):
            prompt = prompData[i]
            completion = completionData[i]

            seperator = "->" # last line of problem02.txt is important for this otherwise this separator becomes too long
            # prompt suffix = '$<n>$ ->'
            endingToken = "##" # last line of solution04.txt is important for this otherwise this separator becomes too long
            # completion suffix = "$<n>$ ##"

            line = f"{{\"prompt\":\"{prompt} {seperator}\", \"completion\":\"{completion} {endingToken}\"}}\n"
            file.write(line)


def getListOfCombinationsFromFileContent(content):
    FUNCTION_NAMES = ["foo", "bar", "main"] # function name
    PARAMS_1 = ["param1"] # first parameter names
    PARAMS_2 = ["param2"] # second parameter names. (For our case there are at most 2 params)
    VARIABLE_1 = ["nthreads"] # variable for number of thread
    VARIABLE_2 = ["output", "result"] # returned variable
    VARIABLE_3 = ["s_priv"] # variable that holds the result computed by each thread
    VARIABLE_4 = ["ARRAY_SIZE", "N"] # variable name for global array size
    VARIABLE_PAD = ["PAD", "PADDING"] # variable used for padding the array for fixing the false sharing
    # <VAR_PAD> is only in solution files. 
    # So, if it has more than one value, there will be more than 1 solution fo each problem.

    tokenDict = {}
    if "<FUNC_NAME>" in content:
        tokenDict["<FUNC_NAME>"] = FUNCTION_NAMES
    if "<PARAM_1>" in content:
        tokenDict["<PARAM_1>"] = PARAMS_1
    if "<PARAM_2>" in content:
        tokenDict["<PARAM_2>"] = PARAMS_2
    if "<VAR_1>" in content:
        tokenDict["<VAR_1>"] = VARIABLE_1
    if "<VAR_2>" in content:
        tokenDict["<VAR_2>"] = VARIABLE_2
    if "<VAR_3>" in content:
        tokenDict["<VAR_3>"] = VARIABLE_3
    if "<VAR_4>" in content:
        tokenDict["<VAR_4>"] = VARIABLE_4
    if "<VAR_PAD>" in content:
        tokenDict["<VAR_PAD>"] = VARIABLE_PAD

    combinations = []
    for valList in tokenDict.values(): # valList is a list of values for a token
        if len(combinations) == 0:
            for i in range(0, len(valList)):
                combinations.append([valList[i]])
        else:
            val = valList[0]
            for j in range(0, len(combinations)):
                combinations[j].append(val)

            for i in range(1, len(valList)):
                temp = deepcopy(combinations) # temp is a list of list
                for j in range(0, len(temp)): # temp[j] is a list of a single combination
                    temp[j][len(temp[j])-1] = valList[i] # replace last element of the combination with new value in valList
                    combinations.append(temp[j])
    return combinations, tokenDict

def handleFiles(fileType):
    counter = 0
    for i in range(1, 16):
        fileName = f"{fileType}{i:02d}.txt"
        
        with open(f"Files\\{fileName}", "r") as file:
            content = file.read()
            
            combinations, tokenDict = getListOfCombinationsFromFileContent(content)

            keyList = list(tokenDict.keys())
            for combination in combinations:
                copyContent = deepcopy(content)
                for i in range(0, len(keyList)): # keyList and combination should be of same size.
                    copyContent = copyContent.replace(str(keyList[i]), combination[i])
                temp = fileName.replace(".txt", "")
                with open(f"AugmentedData\\{temp}_augment{counter:03d}.txt", "w") as oFile:
                    oFile.write(copyContent)
                    counter += 1

def augmentData():
    handleFiles(fileType="problem")
    handleFiles(fileType="solution")


def main():
    FILE_PATH="AugmentedData\\"
    problemCodes = []
    solutionCodes = []
    codeClasses=[]
    # augmentData() # uncomment this line if you want to augment the data in Files directory
 
    for _, _, files in os.walk(FILE_PATH):
        problemCodes = ["" for _ in range(int(len(files) / 2))]
        solutionCodes = ["" for _ in range(int(len(files) / 2))]
        codeClasses = ["" for _ in range(int(len(files) / 2))]

        for fileName in files:  # for each file in input file path
            filePath = os.path.join(FILE_PATH, fileName)
            fileNumber = int(fileName[-7:-4])
            globalFileNumber = int(fileName.split("_")[0][-2:])

            # filling problem and solution codes
            code = ""
            with open(filePath, "r") as file:
                lines = file.readlines()
                for line in lines:
                    code += line.strip() + "$<n>$"

                if "solution" in fileName:
                    solutionCodes[fileNumber-1] = code
                
                elif "problem" in fileName:
                    problemCodes[fileNumber-1] = code
            
            # filling classes
            if 1 <= globalFileNumber and globalFileNumber <= 10:
                codeClasses[fileNumber-1] = "False Sharing"
            else:
                codeClasses[fileNumber-1] = "Loop Unrolling"

    generateJSONL(prompData=problemCodes, completionData=codeClasses, outputFileName="data01.jsonl")
    generateJSONL(prompData=problemCodes, completionData=solutionCodes, outputFileName="data02.jsonl")




if __name__ == "__main__":
    main()
    