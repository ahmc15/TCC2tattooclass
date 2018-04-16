import os
path = 'C:/Users/André/TCC/Banco de Dados/Tavares'
files = os.listdir(path)
a='Tavares'
i = 1

for file in files:
    os.rename(os.path.join(path, file), os.path.join(path, a+str(i)+'.jpg'))
    i = i+1