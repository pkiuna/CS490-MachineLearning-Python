data = input('Enter string:')
count = int(input('Enter number of characters you want to delete:'))

# stroring after the number of characters deleted
res = data[0:-count]
print(res[::-1])

num1 = input('Enter first number:') #ask for input number
num2 = input('Enter second number:')

a = int(num1)
b = int(num2)

print('Addition:',a + b) #perform addition
print('Subtraction:',a-b) #perfom subtraction