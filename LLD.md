
# LLD

LLD dives into the specifics of implementing individual features
It involves low level design like implementing classes, modules, design patterns, it doesn't concern hld like caching, load balancers, api gateways, etc...
THe code should be extensible.

**Steps for LLD:**

- Requirements gathering (user activity, user flow from start (input) to end (output)), determine entities, actions and expected outputs
- Break down the requirements
- Laying Down Use Cases
- UML Diagrams
- Model Problems (Design patterns that can be used)
- Implement Code


## **OOPS**

### Class

A class is a blueprint for creating objects. It has some attributes(features) and methods (functions/behaviours). It's not an actual instance, it's blueprint, the objects createed using a constructor of this class are the instances.

A constructor is used to initialize an instance of a Class type. There are different types of cosntructor like default(no parameter/arguments, so the attributes will have default values like null or 0 or ''), parameterised. 

**Overloading and Overriding:**

**Method overloading** allows multiple methods to have same name, but the number of parameters, type of parameters, or rturn type should be differnt.

On the other hand, **Method overriding** is used in subclass, so same name method exists in parent class and the subclass overrides the method according to its own use case.

**Static** means belong to class, whether an attribute or a method, if static, it belongs to the class not to an instance/object

Constructors can be called using this() for the same class or super() for parent's class 
Constructors can have return keyword to exit, but it doesn't return values.

this keyword helps to reference an object.





## Principles

### SOLID

### DRY

### KISS

### YAGNI
