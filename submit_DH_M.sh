#!/share/apps/opt/python/2.7.9/bin/python2
#$ -S /share/apps/opt/python/2.7.9/bin/python2
#$ -V
#$ -cwd
#$ -j y

#$ -o ./output
#$ -e ./error
#$ -q batch.q

import os

M = [500]
D_hidden = [25, 30]

Ninit = 50

SGE_TASK_ID = int(os.getenv("SGE_TASK_ID", 0))

i_M = (int(SGE_TASK_ID - 1) / (int(Ninit) * int(len(D_hidden)))) % int(len(M))
i_DH = (int(SGE_TASK_ID - 1) / int(Ninit)) % int(len(D_hidden))
initID = int(SGE_TASK_ID - 1) % Ninit + 1
adolcID = SGE_TASK_ID % 2000

print("M = %d"%(M[i_M],))
print("D_hidden = %d"%(D_hidden[i_DH],))
print("initID = %d"%(initID,))
print("SGE_TASK_ID = %d"%(SGE_TASK_ID,))

print(os.system("uname -n"))

#if i_M == 4:
#    print("Skipping M=1000 case.")
#else:
os.system("python2 varnet_mnist.py %d %d %d %d"%(initID, M[i_M], D_hidden[i_DH], adolcID))
