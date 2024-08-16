from turtle import left
import matplotlib.pyplot as plt
import csv
import seaborn as sns
import plotly.graph_objects as go
import sys

print(plt.get_backend())

in_file = sys.argv[1]

def get(strainNumbers): #input = ['1', '1', '2', '2'] : List of number strings
    strainNumbers = strainNumbers[1:-1].replace("'", "").replace(" ", "").split(",")
    newLabel = "G"+strainNumbers[0] + "P"+strainNumbers[1] + "R"+strainNumbers[2] + "C"+strainNumbers[3]
    # newLabel = "G"+strainNumbers[0] + "P"+strainNumbers[1]
    return newLabel 
    
x = []

#colours = {"G1P1":"r", "G2P2":"c", "G1P2":"g", "G2P1":"y"}
#markers = {"R1C1": "+", "R2C1":"o", "R1C2":"x", "R2C2":"*"}
#colors = {("1", "1"):'r', ("1", "2"):'g', ("1", "3"):'b', ("2", "1"):'y', ("2", "2"):'c', ("2", "3"):'m', ("3", "1"):'coral', ("3", "2"):'coral', ("3", "3"):'orange'}

print("reading input")

inputfile = open(in_file)
lines = list(csv.reader(inputfile, delimiter=','))
numStrains = len(lines[0]) - 2
print(numStrains)

ys = [list() for i in range(numStrains) ]

for line in lines[1:]:
    print(len(line))
    x.append(float(line[0]))
    for i in range (numStrains):
        ys[i].append(float(line[i+1]))
print("drawing plots")

pal = list(sns.color_palette(palette='viridis', n_colors=len(ys)).as_hex())

fig = go.Figure()
i = 0
for d,p in zip(ys, pal):
    if any(t>0 for t in d):
        fig.add_trace(go.Scatter(x = x,
                                y = d,
                                name = get(lines[0][i+1]),
                                fill=None))
    i += 1
fig.update_layout(title=in_file)
fig.show()
#fig.write_html('first_figure.html', auto_open=True)

# bar plot for time=200y



# plt.plot(x,immunity1,label="immunity-s1")
# plt.plot(x,immunity2,label="immunity-s2")

# #plt.xlim(0, 1)
# plt.ylim(bottom=0)
# #plt.xlim(left=0)
# plt.xlabel("time")
# plt.ylabel("ImmunedHosts")
# plt.legend()
# plt.tight_layout()
# plt.savefig('C:/Users/suvan/OneDrive/Desktop/ABM/plot2.jpeg')