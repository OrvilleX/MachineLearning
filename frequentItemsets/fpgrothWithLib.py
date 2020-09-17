import pyfpgrowth


def simpleTest():
    transactions = [[1, 3, 4],
                    [2, 3, 5],
                    [1, 2, 3, 5],
                    [2, 5]]
    patterns = pyfpgrowth.find_frequent_patterns(transactions, 2)
    rules = pyfpgrowth.generate_association_rules(patterns, 0.5)
    print(rules)


if __name__ == '__main__':
    simpleTest()