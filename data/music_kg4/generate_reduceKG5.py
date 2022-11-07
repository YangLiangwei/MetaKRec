# 根据给定的weight threshold缩小kg
# 包含阈值！
import sys

if sys.argv[1] == '4':
    kg_in = "data/kg_generate_nx4.txt"
    kg_out = "data/kg_generate4.txt"
elif sys.argv[1] == '5':
    kg_in = "data/kg_generate_nx5.txt"
    kg_out = "data/kg_generate5.txt"
with open(kg_out,"w",encoding="utf-8") as f:
    for line in open(kg_in,"r"):
        line = line.strip('\n')
        line = line.split('\t')
        if int(line[2]) > int(sys.argv[2]):
            f.write(line[0])
            f.write('\t')
            f.write('10000')
            f.write('\t')
            f.write(line[1])
            f.write('\n')

