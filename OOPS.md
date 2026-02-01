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
