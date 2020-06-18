class employee:
    # Members to count employees and find average salary
    counter = 0
    salary = 0

    #contructor to initialize family, salary, department using self
    def _init_(self, name, family, salary_num, department):
        self.name = name
        self.family = family
        self.salary_num = salary_num
        self.department = department

        employee.counter += 1  # count of employees
        employee.salary += salary_num  # Average the salaries of employees

        # calculate average salary
        def get_avg_salary(self):
            avg_salary = self.salary / self.counter
            print("The average salary is: " + str(avg_salary))
            return

        def _init_(self, name, family, salary_num, department):  # inherit of employee class when initializing
            employee._init_(self, name, family, salary_num, department)


def main():
    emp1 = employee("Paul", "Kiuna", 4000, "IT")
    emp1.display()
    emp2 = employee("Rachel", "Morris", 4000, "CS")
    emp2.display()
    emp3 = employee("Matthew", "Stocks", 4000, "IT")
    emp3.display()
    emp4 = employee("Rachel", "Stills", 4000, "CS")
    emp4.display()
    emp5 = employee("Morgan", "Belle", 4000, "IT")
    emp5.display()
    emp6 = employee("George", "Kiuna", 4000, "CS")
    emp6.display()
if __name__ == '__main__':
    main()