with open("results.txt", "r") as results:
    result = [i.strip().split(" ") for i in results.read().split("\n") if len(i) > 1 and i[0] in map(str, range(10))]
tests = ["RecurrentLidarNet", "MLP256S", "MLP256M", "MLP256L", "TinyLidarNetS", "TinyLidarNetM", "TinyLidarNetL"]
tracks = ["GYM", "AUS", "MOS", "SPL"]
lists = []
for i in range(len(result)):
    if i % 10 == 0:
        lists.append([[],[]])
    if result[i][3] in ["LAP", "COLLISION:", "TIMEOUT:"]:
        lists[-1][0].append(float(result[i][-1]))
    if result[i][3] in ["LAP"]:
        lists[-1][1].append(float(result[i][-3][0:-1]))
accuracies = []
times = []
for i in range(len(lists)):
    if len(lists[i][1]) == 0:
        times.append("N/A")
    else:
        times.append(sum(lists[i][1])/len(lists[i][1]))
    accuracies.append(sum(lists[i][0])/len(lists[i][0]))
allresults = "\t\t\tAverage Lap Time (s)\t\tAverage Progress (%)\nModel\t\t\tGYM\tMOS\tAUS\tSPL\tGYM\tMOS\tAUS\tSPL"
results = []
timescounter = 0
accuraciescounter = 0
for i in range(len(times)+len(accuracies)):
    if (i//4)%2 == 0:
        results.append(times[timescounter])
        timescounter += 1
    else:
        results.append(accuracies[accuraciescounter])
        accuraciescounter += 1
for i in range(len(times)+len(accuracies)):
    if i % 8 == 0:
        allresults+="\n\n"+tests[i//8]+"\t"*(3-len(tests[i//8])//8)
    allresults += str(results[i])[0:5]+"\t"
with open("recurrentlidarnetresults.txt", "w") as results:
    results.write(allresults)