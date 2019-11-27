from sklearn import tree

##### CARACTERÍSTICAS #####
animais = {
	'cachorro':0,
	'gato':1,
	'humano':2
	}
som = {
	'lati': 0,
	'mia': 1,
	'fala': 2
	}
locomocao = {
	'quadrupede': 0,
	'bipede': 1
	}
dentes = [42, 30, 32]
# Peso médio
# Cachorro: 15 - 25 Médio Porte
# Gato: 3.6 – 4.5
# Humano: 62 - 80
#### CARACTERÍSTICAS ####

		# [som, locomoção, dentes, peso]
features = [
		[som['lati'], locomocao['quadrupede'], dentes[0], 18],
		[som['lati'], locomocao['quadrupede'], dentes[0], 20],
		[som['mia'], locomocao['quadrupede'], dentes[1], 4.2],
		[som['mia'], locomocao['quadrupede'], dentes[1], 3.8],
		[som['fala'], locomocao['bipede'], dentes[2], 72],
		[som['fala'], locomocao['bipede'], dentes[2], 90],
		]

labels = [
	animais['cachorro'], 
	animais['cachorro'], 
	animais['gato'], 
	animais['gato'], 
	animais['humano'], 
	animais['humano']
	]

clf = tree.DecisionTreeClassifier() # cria o classificador
clf = clf.fit(features, labels) # identifica os padrões

########## ENTRADA ##########
ipt_som = som['fala']
ipt_locomocao = locomocao['bipede']
ipt_dentes = 32
ipt_peso = 75
#############################
resultado = clf.predict([[ipt_som, ipt_locomocao, ipt_dentes, ipt_peso]])
resultado_taxa = clf.predict_proba([[ipt_som, ipt_locomocao, ipt_dentes, ipt_peso]])

if resultado == 0:
	print("Cachorro")
elif resultado == 1:
	print("Gato")
else:
	print("Humano")
print('Precisão: '+str(resultado_taxa[0][resultado][0]*100)+'%')
