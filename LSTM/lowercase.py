lines = []
with open('harrypotter.txt', 'r') as f:
	for line in f:
		lines.append(line.lower())

with open('harrypotter2.txt', 'w') as f2:
	for line in lines:
		f2.write(line)