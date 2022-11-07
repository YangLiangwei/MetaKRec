# 产生reduceKG4数据，对于kg_final中的a-A,看作无向图,A-b,忽略关系。
import networkx as nx
import sys
# 参数1,物品实体的最大编号 last-fm 46184个物品  yelp 45309个物品
nodes=[]
edges=[]
G = nx.MultiGraph()
triple = 0
count = 0
progress = 0
fprint = open('data/log5.txt','w',encoding='utf-8')
# for line in open('kg_final_example.txt'):
for line in open('data/kg_final.txt'):
    count += 1
    line = line.strip()
    data = line.split('\t')
    head = int(data[0])  
    tail = int(data[2])
    relation = int(data[1])
    nodes.append(head)
    nodes.append(tail)
    edges.append((head,tail,relation))
#    if count == 20:
#        break
# print(nodes)
# print(list(set(nodes)))
G.add_nodes_from(list(set(nodes)))
G.add_weighted_edges_from(edges)
with open('data/kg_final_out_5.txt','w') as fi:
    for n in G:
        progress += 1
        if (progress % 5000 == 0):
            fprint.write('搜索了'+str(progress)+'个节点\n')
            fprint.flush()
        if n < int(sys.argv[1]): # 头必须是物品实体
            for k in G[n]:
                if k >= int(sys.argv[1]): # 中间必须是非物品实体
                    for v in G[k]:
                        if v != n and v < int(sys.argv[1]):
                            a = set()
                            b = set()
                            for indx in G[n][k]:
                                a.add(G[n][k][indx]['weight'])
                            for indx in G[k][v]:
                                b.add(G[k][v][indx]['weight'])
                            c = a & b
                            if len(c) != 0:
                                fi.write(''+str(n)+'\t'+str(k)+'\t'+str(v)+'\n') 
                                triple = triple + 1
fprint.write('抽样后还剩多少kg triple: '+str(triple)+' 原始数据集的：'+str(count))
fprint.close()
