from efficient_apriori import apriori


def simpleTest():
    transactions = [(1, 3, 4),
                    (2, 3, 5),
                    (1, 2, 3, 5),
                    (2, 5)]
    itemsets, rules = apriori(transactions, min_support=0.5, min_confidence=0.5)
    print(itemsets)
    print(rules)


if __name__ == '__main__':
    simpleTest()