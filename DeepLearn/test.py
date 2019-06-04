from functools import reduce

input_vecs = [[1,1], [0,0], [1,0], [0,1]]
labels = [1, 0, 0, 0]

samples = zip(input_vecs, labels)
print(samples)
for sample in samples:
    print(sample)

print(list(map(lambda x: x**2, [1,2,3,4,5])))

print(reduce(lambda x, y: x + y, [2, 3, 4, 5, 6], 1))
#结果为21( (((((1+2)+3)+4)+5)+6) )
print(reduce(lambda x, y: x + y, [2, 3, 4, 5, 6]))
#结果为20