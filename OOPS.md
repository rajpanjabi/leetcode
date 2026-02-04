## OOPS


### Java imp pointers

- Strings are immutable i.e they cannot be modified, when we do string concatenation or modify it, a new memory location is created and the modified string is stored there.  


- **Method Overriding**: This allows to rewrite the method in a child class (method name, nos of params and type remain same, the logic may differ). This only happens in case of inheritance

- **Method Overloading**: On other hand, allows us to use same method name within a class but either the  nos of params or data type of these params must be different. No need for inheritance.
e.g (void add(int a, int b), void add(double a, double b), void add(int a, int b, int c))
Methods cannot be differentiated based on the method's return type!! 

- **Super**: This keyword is used in child class to access parent class methods and variables (constructors, or methods or attributes). for instance super.name 

- Any **private** attribute or method in a class cannot be accessed directly in the subclass, we can access it using getters and setters.

- A **static** method (one that belongs to the class not an instance) cannot be **overriden**.
To access this static method from a subclass we do ParentClassName.staticMehtod. Note: super.staticMethod also doesn't work because super is also an instance of parent class, not the parent class itself and as we know static method/attributes belong to class not instances.

- A static method cannot invoke non static method inside it, but a non-static method can incoke a static method. This happens because we need an object to invoke a non-static method, hence it cannot be done from a static method!

- A static block is loaded/executed when a class is loaded, no need for object creation.


### Encapsulation

The main idea behind encapsulation is to hide all important details and behviours(methods) from outer world. We use private access modifier for that, this makes the attributes and methods unaccessible outside the class. To access them we use public getter and setter methods, which helps us keep control over the logic and avoid any mishap. This promotes code readability, maintainability as well as security.




### Inheritance
For a class (child) to extend another class (super or parent), we use a keyword namely extends.
```java
class Shape{
    private String name;
    private String color;

    public Shape(String name, String color){
        this.name=name;
        this.color=color;
    }

    public void draw(){
        System.out.println("Drawing");
    }
    public getName(){
        return this.name
    }
}

class Rectangle extends Shape{
    private int length;
    private int width;

    public Rectangle(String name, String color, int length, int width ){
        super(name,color);
        this.length=length;
        this.width=width;
    }
    public getLength(){
        return this.length;
    }

}

class Main{

    public static void main(String[] args){
        Rectangle rec=new Rectangle();
        System.out.println(rec.name);
        rec.getName();
    }
}

```

- In Java, multiple inheritance is not possible, i.e a class cannot mulitple class together.
- It can either extend a single class, or can extend a class which further extend anotehr class, which is known as multi-level inheritance. There's also hierarchial inheritance

- Multiple inheritance refers to a feature in object-oriented programming where a class can inherit properties and methods from more than one parent class. This allows the child class to combine the functionality of multiple parent classes.

- Diamond Problem: Diamond Problem occurs when a class inherits from two classes that have methods with the same name. The compiler cannot determine which method to execute.

![Alt text](/diamondPrblm.png)

-If both B and C inherit from A and override a method, and D inherits from both B and C, which version of the method should D inherit? This ambiguity is why Java restricts multiple inheritance for classes.

- So what's the alternative to this? Java allows multiple inheritance through interfaces, as interfaces only declare method signatures (no method bodies initially), thus preventing conflicts.


### Polymorphism

The word Poly-morphism breaks as poly(many/different) and morphism(forms). It refers to the ability of a single entity (like a method, operator, or object) to behave differently in different contexts.



There are two main types of polymorphism in Java:

- Compile-Time: For this type, which method to call is decided at compile time itself. This is seen in case of method overloading when multiple methods of same name but diff params are present in class. When we call the method, it can be resolved at the compile time itself that which method needs to be called based on nos of params and their datatype.

- Run-Time: This on the other hand takes place in case of Inheritance or when we have overriden methods. In this scenario when we compile we still don't know which method to call because the object is not created until we run the program. Once the program is run we create instacne of child and parent class both, now we know whether to call the method of parent or child class.


### Abstraction

Process of hiding implementation details (methods) and exposing the required features. Abstraction allows us to focus on what an object does, rather than how it does it.


Abstract methods have no body (i.e., they have no implementation). 


Abstraction is achieved through abstract classes and interfaces


An abstract class is a class that cannot be instantiated on its own and must be inherited by a subclass. Abstract class can have both abstract and non-abstract (concrete) methods.


Interface can also have abstract, default and static methods (after Java 8). Interface needs to be implemented.


```java

abstract class Animal{
    private String legs;

    // abstract method
    public abstract void move(){}

    // concrete method
    public void makeNoise(){
        System.out.println("Noiseee");
    }

    // static method
    public static void test(){
        System.out.println("belongs to class");

    }


}

class Dog extends Animal{
    // need to implement the static method
    @Override
    public void move(){
        System.out.println("Dog moving");
    }


}

```

---

```java

interface Animal {
    void sound(); // Abstract method
    void eat();   // Abstract method
}

// Implementing the interface in the Dog class
class Dog implements Animal {
    @Override
    public void sound() {
        System.out.println("The dog barks.");
    }

    @Override
    public void eat() {
        System.out.println("The dog eats food.");
    }
}

```

- An abstract class can extend another abstract class in Java.

- An abstract class can inherit from another abstract class just like a regular class would. The subclass (child abstract class) will inherit the abstract methods and behaviors of the parent class, but it is not required to implement the abstract methods from the parent class unless it is a concrete class (i.e., a class that is not abstract).

- If the subclass is also abstract, it can either:
    -  Implement the abstract methods from the parent class, or
    -  Leave them unimplemented (in which case, the subclass must also be declared as abstract).

- We cannot create an instance of an abstract class directly in Java. An abstract class is designed to be inherited by other classes, and it is not meant to be instantiated on its own.

- However, an abstract class can have a constructor, which can be invoked by a subclass when an instance of the subclass is created. This allows the abstract class to initialize its fields before the subclass adds its own specific behaviors


#### Interfaces

Interfaces are pure contracts. They only had abstract methods and abstract final variables (constants), but after Java 8, they introduced default methods that helps in backward compatibility i.e if a new method is introduced in interface it will not break the logic of all the subclasses that implements this interface, we can add logic in this default method which can be used by all these sub-classes.

A class can implement multiple interfaces.
Interfaces cannot have constructors.
Interface can extend (inherit) from another interface. 
All variables in an interface are implicitly public, static, and final. This means they act as constants and cannot be changed.





### Inner Class

Any java class which is defined inside another class is known as Inner Class. There are 4 main types of inner class:

1) **Static Nested Inner Class**:
    A static nested class is defined with the static modifier. Since it is static, it does not require an instance of the outer class to be created. Static nested classes can only access the static members of the outer class.

    ```java

    class Outer{
        static int cnt=0
        String name="Outer";


        static class Inner{
            void display() {
                System.out.println("Static variable: " + staticVar);
            }


        }
    }
    class Main{

        public static void main(String[] args){
            Outer.Inner inner = Outer.new Inner();
            inner.execute();
        }
    }
    ```



2) **Non-Static Inner Class**:
    A non-static inner class is associated with an instance of the outer class. It has access to all members (both static and non-static) of the outer class, including private members.
    
    ```java
    class Outer{
        static int val=10;
        String name="Outer";


        class Inner{
            public void execute(){
                System.out.println("I can access static and non static values like val: " +val +" name: "+name);
            }
        }
    }

    class Main{

        public static void main(String[] args){
            Outer outer = new Outer();
            Outer.Inner inner= outer.new Inner();
            inner.execute();
           // this prints both static and non static values in print statement
        }
    }

    
    ```

3) **Local Inner Class**:
    Local inner classes are defined within a method or a block of code. They are only accessible within the scope of that method or block. Local inner classes can access all members of the outer class but can only access the effectively final local variables of the enclosing method. i.e anything from Outer class, but can only access the variables in that block that are final/are not changed once instantiated.

    ```java
    class Outer{
        static int val=10;
        String name="Outer";

        public void outerMethod{
            int test=3
            class LocalInnerClass{
                void display(){
                    int cnt=0
                    test+=2;
                    // here I can access anything of outer class, but the cnt of the method can only work if it not changes at any point in the method block
                    System.out.println("val: "+val+" name: "+name);
                    System.out.println("cnt works: "+cnt);
                    // test wouldn't work because it changes in the mehtod and is not final
                    
                }
            }
        }
    }
    class Main{

        public static void main(String[] args){
            OuterClass outerObj = new OuterClass();
            outerObj.outerMethod();
        }

    }
    ```

4) Anonymous Inner Class:
    Anonymous inner classes are a type of local inner class without a name. They are often used to implement interfaces or extend classes for one-time use.

    ```java
    abstract class Greeting {
    abstract void sayHello();
    }       

    class Main {
        public static void main(String[] args) {
            Greeting greeting = new Greeting() { // Anonymous inner class
                void sayHello() {
                    System.out.println("Hello, World!");
                }
            };

            greeting.sayHello(); // Output: Hello, World!
        }
    }
    ```


### Association, Aggregation and Composition

How does objects interact with each other is the relationship they have with each other.

#### Association: A general relationship where one class interacts with another.
    It represents a situation where objects of one class interact with objects of another through some form of linkage or reference. This interaction can manifest in various ways, depending on the context of the relationship.
    The connection can be one-to-one, one-to-many, or many-to-many, enabling different levels of collaboration and data sharing between the classes. 


    Types of Association:

        1) One-to-One: One instance of a class is associated with exactly one instance of another class. For example, a Person class might have a one-to-one relationship with a Passport class.

        2) One-to-Many: One instance of a class is associated with multiple instances of another class. For instance, a Teacher class may be associated with multiple Student objects.

        3) Many-to-Many: Many instances of a class are associated with many instances of another class. For example, a Student class might be associated with multiple Course objects, and each Course object can have multiple Students
    
```java
    // One-to-One
    class Passport{
        private String passportNumber;
        public Passport(String passportNumber){
            this.passportNumber=passportNumber;
        }
        public String getPassportNumber(){
            return passportNumber;
        }

    }

    class Student{
        private String name;
        private Passport passport;
        public Student(String name, Passport passport){
            this.name=name;
            this.passport=passport;
        }
        public String getName(){
            return name;
        }
        public void displayDetails(){
            System.out.println("Student: "+ name);
            System.out.println("Passport:" + passport.getPassportNumber());
        }

    }

    class Main{
        // Here, there is one-to-one relationship between Student and Passport object. 
        // One Student can have only one passport and vice versa.
        public static void main(String[] args){
        // We need to create passport and student object separately
        
            Passport passport = new Passport("12345678");
            Student student = new Student ("Raj", passport);
            student.displayDetails();

    }
    }

```

```java
    // Many-to-Many
    //School and Student
    import java.util.ArrayList;

    class School{
        private String name;
        private ArrayList<Student> students;
        public School(String name){
            this.name = name;
            this.students = new ArrayList<Student>();
        }

        public void addStudent(Student student){
            students.add(student);
        }
        public void listStudents(){
            for (Student st : students){
                System.out.println("Student: "+ st.getName());
            }
        }


    }

    class Student{
        private String name;
        public Student(String name){
            this.name=name;
        }
        public String getName(){
            return this.name;
        }

    }

    class Main{

        public static void main (String[] args){
            Student ram = new Student("Ram");
            Student rahul = new Student("Rahul");
            School school1 = new School("school1");
            school1.addStudent(ram);
            school1.addStudent(rahul);
            school1.listStudents();
        }

    }

```


Similarly for Many-to-Many, we can take example of Multiple students can enrol in different courses. i.e one course can have multiple students and similarly one student can enrol in many courses.
 


#### Aggregation is a specialized form of association.
    It is a type of relationship which is loosely coupled. Both the entities can exist independently.
    It represents a "whole-part" relationship where the "whole" and "part" can exist independently. For example, a Department class may contain multiple Employee objects, but the employees can exist independently of the department.

```java
    class Department{
        private String name;
        private ArrayList<Employee> employees;
        public Department(String name){
            this.name=name;
            this.employees=new ArrayList<Employee>();
        }

        public void addEmployee(Employee employee){
            employees.add(employee);
        }
        public void displayEmployees(){
            for (Employee emp : employees){
                System.out.println("employee: "+ emp.getName());
            }
        }


    }

    class Employee{
        private String name;
        public Employee(String name){
            this.name=name;
        }
        public String getName(){
            return this.name;
        }

    }

    class Main{ 
        public static void main( String[] args){
            // Creating employees
            Employee emp1 = new Employee("emp1");
            Employee emp2 = new Employee("emp2");
            Employee emp3 = new Employee("emp3");
            // Creating Dept
            Department dept = new Department("HR");
            // adding employee to dept
            dept.addEmployee(emp1);
            dept.addEmployee(emp2);
            dept.addEmployee(emp3);
            // displaying employees of dept
            dept.displayEmployees();

            // Notice here, if we delete department, employees don't stop existing , they are still here.
            // this is a weakly coupled relationship known as aggregation where employees and department exist independently
        }


    }

```

#### Composition 

    Composition is a stricter form of aggregation where the "whole" and "part" are tightly coupled. If the "whole" is destroyed, the "parts" are also destroyed. This represents a "part-of" relationship. For example, a House class might contain multiple Room objects. If the House is destroyed, the Room objects cease to exist.

```java
import java.util.*;
class House {
    private List<Room> rooms;

    public House() {
        rooms = new ArrayList<>();
        rooms.add(new Room("Living Room"));
        rooms.add(new Room("Bedroom"));
    }
}

class Room {
    private String name;

    public Room(String name) {
        this.name = name;
    }
}

// if we carefully observe here we are creating the part(rooms) inside the whole(house), which makes them tightly coupled, as soon as the whole(house) is destroyed, the part(rooms) is also destroyed indicating that they cannot exist independently.


```


### Object Cloning

Cloning refers to copying an object. When we clone an object, we mainly just copy the primitive fields completely, but for objects we just copy the reference to the object. i.e for instance if a Student class has name and a Passport object inside it, wehen we create a clone of Student,w e copy the field name and the reference pointer of passport object(not the object itself), this is known as shallow copy/cloning. So if we modify the referenced object(passport in this case), the change is reflected on both the cloned and original object. 

- Shallow cloning creates a new object that is a duplicate of the original object but only at the surface level. The new object will have the same values for all primitive fields, and references to the same memory locations for any reference-type fields (e.g., objects, arrays, or collections). This means that while the cloned object is distinct from the original, any modifications to shared references will be reflected in both objects. Shallow cloning is done using clone() method.

- Deep cloning ensures that a completely independent copy of the object is created, including all nested objects. This prevents unintended modifications in the original object when the cloned object is modified. Since the default clone() method performs a shallow copy, deep cloning requires manual cloning of all referenced objects. This can be done using the clone() method recursively.


- The clone() method is defined in the Object class. By default, it performs a shallow copy of the object. When invoked, it:
        Allocates a new memory location for the cloned object.
        Copies the field values (primitives are copied, references are copied but not the objects they refer to).

- To make a class/object clonable, we must make sure the class implements Clonable interface. Since this is an interface, the class must override the clone method and add logic where it creates a new object of the class type and assigns it with super.clone() and finally returns it. This is shallow copy. For deep cloning, we must do this process for each of the object inside this class and have those classes implement clonable interface as well.

- Also another imp thing is that the overriden clone method should also throw CloneNotSupportedException.


```java
// Shallow cloning

import java.util.*;
// Address class
class Address {
    String city;
    
    // Constructor
    Address(String city) {
        this.city = city;
    }
}

// Person class (which is clonable)
class Person implements Cloneable {
    String name; // Primitive field 
    Address address; // Reference-type field

    // Constructor
    Person(String name, Address address) {
        this.name = name;
        this.address = address;
    }
    
    // clone() method is inherited from Object class and must be Overriden 
    @Override
    protected Object clone() throws CloneNotSupportedException {
        return super.clone();  // Shallow copy
    }
}

class Main {
    public static void main(String[] args) throws CloneNotSupportedException {
        Address address = new Address("Mumbai");
        Person person = new Person("Rahul", address);

        Person clonedPerson = (Person) person.clone(); // Cloning person

        // Modifying the address in the cloned object
        clonedPerson.address.city = "New Delhi";

        // Output to check if changes are reflected in the original
        System.out.println(person.name + " lives in " + person.address.city);  // New Delhi
        System.out.println(clonedPerson.name + " lives in " + clonedPerson.address.city);  // New Delhi
    }
}

```


```java
//Deep cloning
import java.util.*;
// Address class (which is cloneable)
class Address implements Cloneable {
    String city;
    
    // Constructor
    Address(String city) {
        this.city = city;
    }
    
    // Overriding default clone() method 
    @Override
    protected Object clone() throws CloneNotSupportedException {
        return new Address(this.city);  // Creating a new object
    }
}


// Person class which is cloneable
class Person implements Cloneable {
    String name; // Primitive field
    Address address; // Reference-type field
    
    // Constructor
    Person(String name, Address address) {
        this.name = name;
        this.address = address;
    }
    
    // Overriding
    @Override
    protected Object clone() throws CloneNotSupportedException {
        Person clonedPerson = (Person) super.clone(); // Shallow copy
        
        // Cloning nested object for Deep Cloning 
        clonedPerson.address = (Address) address.clone(); 
        return clonedPerson;
    }
}

class Main {
    public static void main(String[] args) throws CloneNotSupportedException {
        Address address = new Address("Mumbai");
        Person person = new Person("Rahul", address);

        Person clonedPerson = (Person) person.clone(); // Deep Cloning

        // Modifying the address in the cloned object
        clonedPerson.address.city = "New Delhi";

        // Output to check if changes are reflected in the original
        System.out.println(person.name + " lives in " + person.address.city);  // Mumbai
        System.out.println(clonedPerson.name + " lives in " + clonedPerson.address.city);  // New Delhi
    }
}

```



### Exception Handling

Exception handling is a mechanism in Java that helps manage runtime errors and maintain the normal flow of a program. An exception is an unwanted or unexpected event that occurs during program execution, disrupting the normal flow. Java provides a robust exception handling framework to catch and handle such situations efficiently.

Mainly two types of exceptions: one that happens during compile time (a.k.a Checked), one that is identified during runtime (a.k.a Unchecked).

Exception handling is very important in order to avoid the chances of breaking the app. It also helps in debugging, encapsulating error handling logic and also ensures proper resource management i.e prevents memory leaks by ensuring files and connections are properly closed.

The best way to handle exception is using a try-catch block. It comprises of one try block and multiple catch blocks.It can also consist of a finally block which is executed regardless there is an error or not. (This finally block is mostly used to close resources such as files, database connections, or network sockets. , etc to free up resources, once task is performed or an error is encountered)


try block: Contains the code that may throw an exception.
catch block: Handles the exception if it occurs.



```java

import java.util.*;

class Main {
    public static void main(String[] args) {
        int num1 = 10, num2 = 0;
        
        // Exception handling
        try {
            int result = num1 / num2; // Risky code, runtime exception
            System.out.println("Result: " + result);
        } 
        // inside catch we can mention any of the Exception module 
        catch (ArithmeticException e) {
            System.out.println("Error: Division by zero is not allowed!");
        }
        // Catch block to handle ArrayIndexOutOfBoundsException
        catch (ArrayIndexOutOfBoundsException e) {
            System.out.println("Error: Array index out of bounds!");
        } 
        // finally block 
        finally {
            System.out.println("Finally block executed.");
        }
        
        // Remaining code 
        System.out.println("Program continues...");
    }
}


```

#### Throw and Throws

Java provides two important keywords — throw and throws — for handling exceptions effectively. While they might sound similar, they serve different purposes in exception handling.

1) throw - Manually Throwing an Exception
    The throw keyword is used inside a method or block to explicitly throw an exception.
    It is typically used when we want to indicate that an error has occurred due to invalid input or some exceptional condition.
    The throw statement is followed by an instance of an exception.


```java
import java.util.*;
throw new ExceptionType("Error Message");

//Note that The ExceptionType must be a subclass of Throwable (like ArithmeticException, NullPointerException, or a user-defined exception).

```

2) throws - Declaring an Exception
    The throws keyword is used in a method signature to indicate that the method might throw an exception.
    It does not handle the exception but forces the caller to handle it.
    This is useful when a method relies on external resources like files, databases, or network connection.

    Here the method does not handle the exception internally; it leaves the responsibility to the caller.

```java
import java.util.*;
returnType methodName() throws ExceptionType {
    // Method code that might throw an exception
}
```


#### Custom Exceptions

In Java, custom exceptions (also known as user-defined exceptions) allow developers to define their own exception classes. This is useful when the built-in Java exceptions (ArithmeticException, IOException, etc.) do not fully describe the error conditions specific to an application.


```java

class CustomException extends Exception{
    // Constructor
    public CustomException(String message){
        super(message);
    }
}
// Main class
class Main {
    public static void main(String[] args) {
        
        // try-catch ladder
        try {
            // throwing custom made exception
            throw new CustomException("This is a custom exception!");
        } 
        catch (CustomException e) {
            System.out.println("Caught: " + e.getMessage());
        }
        
        System.out.println("Program continues...");
    }
}

```
```

                    Throwable
                        |
                        |
                    Exception
                        |
                        |
            Custom--------------Built-in Exception 
                                    |
                                    |
                        Runtime----------Compile-time
```

### Generics

Generics in java helps to ensure type safety and code reusability.
They allow defining classes, interfaces, and methods that can operate on various data types without sacrificing type safety.


1) Generic Class: A Generic Class works with different types without rewriting the code.

```java
import java.util.*;
class ClassName<T> {
    // T is a type parameter
}

```


2) Generic Methods: A generic method allows different data types within a single method.

```java
import java.util.*;
class ClassName {
    <T> void methodName(T param) {
        // Generic Method
    }
}
```


Bounded type parameters restrict the type of values that can be used as generic arguments, ensuring type safety and enabling operations specific to that type.


```java
import java.util.*;
class ClassName<T extends SomeClass> {
    // Only classes extending SomeClass are allowed as T
}

```




### File Handling

Often times when we build an application we encounter files. We want to read as well as write data to files right? File handling is mostly used for logging, config management(reading data from file, writing to it), data storgae.


Java provides the **File class** from the **java.io** package to handle file operations.

```java
// imports
import java.util.*;
import java.io.File;
import java.io.IOException;


public class Main{
    public static void main( String[] args){
        // create a file 
        try{
            File file =new File("example.text");
            if (file.createNewFile()) {
                System.out.println("File created: " + file.getName());
            } else {
                System.out.println("File already exists.");
            }

        }catch(IOException e){
            System.out.println("An error occurred.");
            e.printStackTrace();

        }

    }
}
```

#### Methods provided in File Class

- createNewFile(): Creates a new empty file.
- exists(): Checks if a file exists.
- delete(): Deletes a file.
- getAbsolutePath(): Returns the file's absolute path.
- length(): Returns the size of the file in bytes.
- canRead(), canWrite(): Checks file permissions.


The **FileWriter** class is used to write character-based data to a file, and **BufferedWriter** improves efficiency by **buffering large amounts of data before writing**.


```java
import java.io.File;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;


class Main{
    public static void main(String[] args){
        BufferedWriter writer= new BufferedWriter(new FileWriter("example.txt"));
        try{
            writer.write("Hello sample text");
            writer.write("part-2");

            // always close the resources
            writer.close();
        }catch(IOException e){
            System.out.println(e);
        }

    }
}

```


The **FileReader** class is used to read data from a file as a stream of characters, while **BufferedReader** improves efficiency by **reading large chunks of data at once**.


```java

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class Main{
    public static void main (String[] args){
        // File file = new File("example.txt");
        try{
            BufferedReader reader=new BufferedReader(new FileReader("example.txt"));
            System.out.println(reader.readLine());
            reader.close();
        } catch (IOException e){
            System.out.println(e);
    }
}
}
```

The **try-with-resources** statement automatically closes the file resource after execution, ensuring proper resource management. This eliminates the chances of missing closing any file used in the codebase preventing resource leakage.

The try-with-resource is nothing but a try block with the resources declared in its arguments.
```java
// Try with Resources (removes the need to close the file explicitly)
        try (BufferedReader reader = new BufferedReader(new FileReader("example.txt"))) { 

        }catch(Exception e){

        }  
```


#### Logging to a File


```java
import java.util.*;
import java.io.*;

// Logger class
class Logger {
    private String path; // to store the path of file
    
    // Constructor
    Logger(String path) throws IOException {
        File file = new File(path); // Open the file path 
        
        // Create the file if it does not exist
        if (!file.exists()) {
            file.createNewFile();
        }
        this.path = path;
    }
    
    // Log the message in the file
    public void log(String message) {
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(path, true))) {
            bw.write(message); 
            bw.newLine();
        } catch (Exception e) {
            System.out.println("Failed to log this message " + message);
        }
    }
}

// Main class
class Main {
    public static void main(String[] args) {
        try {
            // Create a Logger instance with a specified log file path
            Logger myLogger = new Logger("application.log");

            // Log some messages
            myLogger.log("Application started...");
            myLogger.log("User logged in.");
            myLogger.log("Error: Unable to connect to the database.");
            myLogger.log("Application closed.");

            System.out.println("Logs have been written successfully.");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}

```