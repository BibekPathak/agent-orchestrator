def compound_interest(principal, rate, time, n):
    amount = principal * (1 + rate / n) ** (n * time)
    return amount - principal

principal = 1500
rate = 3.5
time = 7
n = 12
print(f'Compound interest: {compound_interest(principal, rate, time, n):.2f}')