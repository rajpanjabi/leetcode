# LLD

LLD dives into the specifics of implementing individual features
It involves low level design like implementing classes, modules, design patterns, it doesn't concern hld like caching, load balancers, api gateways, etc...


It's a crucial phase in the software development lifecycle that focuses on the detailed design of individual components or modules of a system.


It involves specifying the internal structure, algorithms, and data structures that will be used to implement the system's functionality. It also acts as a bridge between high-level design and actual coding.

**Steps for LLD:**

- Requirements gathering (user activity, user flow from start (input) to end (output)), determine entities, actions and expected outputs
- Break down the requirements, functional and non-functional requirements
- Laying Down Use Cases
- UML Diagrams
- Define class structures and diagrams (classes, interfaces, methods)
- Model Problems (Design patterns that can be used)
- Implement Code
- Handle edge cases


## Principles

### DRY
    Don't Repeat Yourself!
    Avoid duplication of logic or code. Repeating code makes the system hard to maintain and error-prone. If a change is required, you might forget to update all occurrences.
    
    How to apply:
    - Identify repetitive code and replace it with a single, reusable code segment.
    - Extract common functionality into methods or utility classes.
    - Leverage libraries and frameworks when available (no need to reinvent the wheel).
    - Refactor duplicate logic regularly across classes or layers

    When not to apply DRY:
    - Premature Abstraction: Don't extract common code too early.
    Extracting similar methods into a shared method can create unnecessary coupling between unrelated parts.
    - Performance-Critical Code: Don't apply DRY to performance-sensitive code if it causes inefficiency.
    Sometimes, repeating optimized low-level logic is faster than calling a generalized, reusable method.
    - Sacrificing Readability: If extracting repeated code makes the code less readable, prefer clarity over DRYness.
    - Legacy Codebases: Don't refactor for DRY's sake in legacy code unless necessary and well-tested.


### KISS
    Keep It Simple, Stupid!
    This principle states that **simplicity** should be a key goal in design and unnecessary complexity should be avoided. In simple terms, use the simplest possible solution that works. Avoid clever, convoluted code.
    
```java
    public static boolean isEven(int number) {
        // Using unnecessary logic to determine evenness
        boolean isEven = false;

        if (number % 2 == 0) {
            isEven = true;
        } else {
            isEven = false;
        }

        return isEven;
    }
```

```java
    public static boolean isEven(int number) {
        return number % 2 == 0;
    }
```

### YAGNI

    This principle states that "Always implement things when you actually need them, never when you just foresee that you need them.". In simple terms, don't add functionality until it's necessary. Avoid building features that you think you might need in the future. This principle helps to keep the codebase clean and reduces unnecessary complexity.


### SOLID

1) Single Responsibility Principle (SRP):

    A class should have only one reason to change. In other words, a class should only have one job, one responsibility, and one purpose.


    If a class takes more than one responsibility, it becomes coupled. This means that if one responsibility changes, the other responsibilities may also be affected, leading to a ripple effect of changes throughout the codebase.


    For instance, if we have a compiler, it does multiple things, checks syntax, run test cases, save outputs to db, return results. In this case, if we break all these things to different classes each handling one thing, it would make the code more readable, maintainable, easier to debug, scale and resusable.

2) Open Closed Principle (OCP):

    Software entities(classes, modules, functions) should be open for extension, closed for modification.




3) Liskov Substitution Principle


















### UML
Unified modeling language

Class Diagram essentials


















### Design Patterns

Design patterns are a foundational concept in software engineering, especially when building scalable and maintainable systems.


Design patterns are standard, time-tested solutions to common software design problems. They are not code templates but abstract descriptions or blueprints that help developers solve issues in code architecture and system design.


Think of design patterns like recipes in cooking. If you want to bake a cake, you don't experiment from scratch each time — you follow a proven recipe. Similarly, design patterns are tried-and-tested “recipes” for solving common coding problems efficiently and consistently.


Mainly 3 broad categories of design patterns:
1) Creational: These focus on object creation mechanisms, trying to create objects in a manner suitable to the situation (5 subtypes)

2) Structural: These deal with object composition — how classes and objects can be combined to form larger structures while keeping the system flexible and efficient. It helps systems to work together that otherwise could not because of incompatible interfaces. (6 subtypes)

3) Behavioral: These are concerned with object interaction and responsibility — how they communicate and assign responsibilities while ensuring loose coupling. (11 subtypes)

### Dependency Injection

In software design, a dependency refers to any object that a class needs in order to function properly.

This technique allows developers to manage the relationships between various components of a system in a more flexible and efficient manner. 

    Dependency Injection (DI) is a design pattern in which an object receives its dependencies from an external source rather than creating them itself.

    In simple terms, instead of creating objects directly inside a class, you "inject" them from outside.

```java
// Without dependency injection
class OrderService {
    private InventoryService inventory = new InventoryService();
    private PaymentService payment = new RazorpayPayment();
    private NotificationService notification = new NotificationService();

    public void checkout(Order order) {
        inventory.blockItems(order);
        payment.process(order);
        notification.sendConfirmation(order);
    }
}
```


Here we are fixing the inventory, payment and notification service i.e whenever we want to use checkout method of the orderService class we have fixed the types of inventory, payment and notification service. 

- If any of this service fails, our orderservice also fails, also if we plan to use any other serivce provider (like stripe), then we'll have to make a lot of changes in existign code (which is not a good practice). 

- This violates SRP and OCP. This is also difficult for testing, i.e in unit testing, we will have to use these as is as opposed to using mock services to just chekc the functionality of orderservice.


There are 3 main ways of using Dependency Injection:

1. Constructor Injection: This is the most commonly used form of Dependency Injection. In this approach, dependencies are passed to the class via the constructor. 

This ensures that the class is always instantiated with its required dependencies, which makes it easier to manage and test.

Once the dependencies are injected through the constructor, they cannot be changed.


2) Setter Injection: In Setter Injection, dependencies are passed to the class via setter methods after the object has been created. 

This allows for mutable dependencies, meaning you can change the dependencies of the class at runtime. Dependencies can be set or changed at any time after the object is created, which gives more flexibility in some scenarios.

One drawback is that the dependencies may not be properly set if the setter method is not called. This can lead to situations where a class is not fully initialized.

3. Interface Injection: In Interface Injection, the dependency provides an injector method that will inject the dependency into the class. This type is rarely used in practice and is typically only suitable for very specific cases.

Example of how to use DI:

```java
// ── Contract: defines what the client needs, not how it is done
interface NotificationService {
    void send(String message);
}

// ── Concrete implementation of the contract
class EmailNotificationService implements NotificationService {
    @Override
    public void send(String message) {
        System.out.println("Email sent: " + message);
    }
}

// ── Client that depends on the abstraction, not the implementation
class UserService {
    // Dependency held as an interface, promoting loose coupling
    private final NotificationService notificationService;

    // Constructor Injection: forces the caller to supply the dependency up-front
    public UserService(NotificationService notificationService) {
        this.notificationService = notificationService;
    }

    // Business logic uses the injected service
    public void register(String user) {
        System.out.println("User registered: " + user);
        notificationService.send("Welcome " + user);
    }
}

// ── Composition Root: the only place where “new” keywords appear
class Main {
    public static void main(String[] args) {
        // Create the concrete dependency
        NotificationService service = new EmailNotificationService();

        // Inject it into the client
        UserService userService = new UserService(service);

        // Execute business operation
        userService.register("raj");
    }
}
```

Key benefits of DI:
Advantages of Dependency Injection (DI)
Dependency Injection brings several advantages to software design. Here are the key benefits of using DI:

1. Swappable Components
Since the dependencies are injected rather than hardcoded, you can easily replace one component with another.
For instance, switching from one payment service to another (e.g., from Razorpay to Stripe) becomes straightforward without changing the core logic of the OrderService.

2. Testable with Mocks
With DI, you can inject mock implementations of your services when running tests. This makes unit testing easier because you don't need to call external services (e.g., payment APIs or database operations). You can replace them with mock objects that simulate their behavior.

3. Follows Dependency Inversion Principle (D in SOLID)
Dependency Injection adheres to the Dependency Inversion Principle (DIP), one of the key principles of SOLID design. DIP states that high-level modules should not depend on low-level modules; both should depend on abstractions. DI enables this by allowing classes to depend on abstractions (interfaces) instead of concrete implementations.

4. Open to Extension, Closed to Modification
This is part of the Open/Closed Principle (O in SOLID), which DI supports. By injecting dependencies, your code is open to extension (e.g., adding new payment types, notification systems, etc.) without modifying existing classes. You can extend the system by simply adding new implementations, without touching the core code.


### Design Patterns

There are 3 main categories of Design Patterns:

1) Creational: These patterns mainly cater to the object creation process of object based on requirements and the scenario. There are 5 sub-types of creational design pattern:
    1) Singleton: This pattern is used when we wish to create a single object of the Class. For instance, when dealing with DB Connections, loggers, etc. It can be implemented lazily or greedily. In Lazy loading, we save computation and resources but there can be problem with concurrency (so use synchronized keyword). In eager loading, we create the single instance/object of class in the constructor itself. 

    2) Factory: This pattern is used when one class deals with multiple different types of object. For example, a Checkout Service class which can use any of the multliple payment gateways. We delegate the object creation process of the payment gateway to the factory and use the interface in the checkout class. When the payment gateway is needed we use factory to create gateway and we use it.

    3) Builder: We mainly use this pattern when we have an object with a lot of fields. For example, a burger class with various attributes like burgerType, bunType, hasSide, hasDrink, list of toppings, etc. An alternate way is to create multiple constructors with different combinations (a.k.a Constructor overloading) , this works in cases with 3-4 fields, but post that it becomes difficult and code readibility is degraded. In builder pattern, we create a builder class (static inner class), we make the burger/object constructor private and we can only create the object using the builder object. 

    4) Prototype: As the name suggests, we have a prototype and we can use it by cloning mulitple copies of it. It is mainly used when object creation process is very expensive and tedious. In such case, we just have a base object and then we create clones out of it. To make sure this works, our object should implement the Cloneable interface and the object class implements the clone method (nested recusrion if required)

    5) Abstract Factory Method: 


2) Structural
3) Behavioural:



### Multi-threading and Concurrency 

Let's say I press play button on Netflix, there are multiple tasks that are happening in the background like loading the video, buffering the content, etc. If each task took 5-6 seconds and happened sequentially, it would add up and take a lot of time. If these tasks could happen at the same time, we would save a lot of time. That’s where concepts like **Multithreading** and **Concurrency** come in


Understanding how to execute tasks efficiently is the key to building fast, responsive systems. 


#### Program, Process and Thread

Lets' take an analogy of a bakery to understand these terms better.

Program: Think of it as a recipe book. It doesn't do anything on its own, it just contains a collection of rules (recipes in this case) i.e a static set of instructions to be followed. 


---


Process: Now, lets say we decide to bake a cake. We pick a recipe from the book and follow the instructions. The baking process is the **process**. A process is a running instance of a program, actively working in memory


A process(heavyweight) includes the program code, current activity, and other resources like memory, CPU usage, and input/output.


Each process has its own dedicated memory space, isolated from others.


Processes are fully isolated from each other, ensuring that one process cannot directly interfere with another.


Communication between processes (e.g., Inter-Process Communication or IPC) is more complex and requires mechanisms like sockets or shared files.
---

Thread: Inside your bakery, there may be multiple bakers working at the same time. One might be mixing ingredients, another might be preheating the oven, and another might be icing the cake. Each baker represents a thread: **a smaller task within the overall process**. Multiple threads can run concurrently, each handling part of the job.


A thread(lightweight) is the **smallest unit of execution** within a process. A process can contain **multiple threads**, which share the same resources but run independently. Each thread can perform a separate task within the same process. Threads allow for parallelism, where multiple tasks are executed simultaneously.


Within the chrome.exe process, there might be several threads running concurrently. For instance, one thread might be responsible for rendering the UI (user interface), another for managing network requests, and another for handling user inputs like clicks or key presses. 


Threads are not isolated; they can directly communicate and share data with other threads in the same process.


#### Cores in CPU

A core in a CPU is a physical processing unit capable of executing instructions. Modern CPU's have multiple cores, allowing them to handle several tasks simulataneously.


Each core can independently execute a thread, meaning more cores lead to the ability to run more threads concurrently, thus improving performance and speed.


#### Hyperthreading

Hyperthreading is a technology developed by Intel that allows a **single physical core to act as two logical cores**. It enables one core to run two threads simultaneously, effectively doubling the number of threads the CPU can handle.


Time slicing refers to dividing the core’s time between multiple threads, ensuring both threads get execution time without wasting resources.


This is done intelligently, so when one thread is waiting for data or performing a slower operation (like I/O), the other thread can continue executing, making better use of the core's resources.


Both threads running on a single physical core share resources like the cache, execution units, and memory bandwidth.


#### Context Switching

Context Switching is the process of storing and restoring the state (or context) of a thread or process so that it can be resumed later. This allows the CPU to switch between different tasks or threads without interrupting their execution entirely, giving the illusion of parallelism, even on a single-core system.


#### Thread Scheduler

The thread scheduler is the part of the operating system that manages context switching. It decides when a thread should be paused and another should be run.


It uses scheduling algorithms (like round-robin, priority-based scheduling, etc.) to determine which thread should run next, triggering a context switch as necessary.


#### Multithreading

Multithreading is a programming technique that allows a CPU to **execute multiple threads concurrently**, with each thread being the smallest unit of a process. It enables a program to perform more than one task at a time within the same process.


Instead of executing one task after another, multithreading allows the CPU to switch between tasks quickly, creating the illusion that multiple tasks are being performed simultaneously.


In simpler terms, multithreading breaks a program into smaller parts (threads) that can run in parallel. Each thread runs independently, but they share the same memory space, which allows them to communicate with each other and work together.


#### Concurrency vs Parallelism

**Parallelism**: It is simultaneous execution of tasks, usually across multiple cores or processors.(like 8 tasks across 8 CPU's, 9 tasks cannot be done parallely on 8 core)


In parallelism, tasks run at the same time, with no context switching required.


**Concurrency**: It involves managing multiple tasks over time, but not necessarily at the same time. It can run on a single core by switching between tasks rapidly, managed by efficient context switching.


For instance, performing 5 tasks on 2 core CPU, first two tasks are run parallely on these cores, as soon as one of the core is free, other task is taken from the queue. In the middle if the task is waiting for something and the core is idle, it can process other task using context switching. This illusion gives us the feeling that things are happening in parallel.



#### Implementing Multithreading

1) By Extending the Thread Class:

The thread class is used to represent a thread in java. We create a subclass that extends this Thread class and overrides the run method. By default this method has a void return type and this is the method that is executed when a thread of the subclass is created. So we put the logic of what we want to be executed for this thread when it is instantiated.


To actually start a new thread of this subclass we use the method start() which under the hood runs the logic from the overriden run method.


join() method: This method makes the main thread wait for the thread to finish before proceeding.

```java
import java.util.*;

class EmailThread extends Thread{
    @Override
    public void run(){
        // logic goes here
        try{
        Thread.sleep(3000);
        System.out.println("Email sent");
        }catch(Exception e){
            e.printStackTrace();
        }
    }
}

class Main{
    public static void main(String[] args){
        EmailThread emailThread= new EmailThread();
        emailThread.start() // this starts the thread and logic from run method

    }
}

```


2) By Implementing a Runnable Interface

Here the runnable interface also has a method run which needs to be overriden by the subclass that implements it. Here we have a runnable object, to create a thread we need to manually pass this runnable instance to thread and then create a thread.

```java
class EmailRunnable implements Runnable{
    @Override
    public void run{
        try {
            Thread.sleep(3000); // 3-second delay for Email
            System.out.println("Email Sent using Runnable.");
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
}

class Main{
    public static void main(String[] args){
        EmailRunnable emailrunnable=new EmailRunnable();
        // Now we need to pass this runnable to the Thread class
        Thread emailThread = new Thread(emailrunnable);
        // to start again use the classic start method
        emailThread.start();
    }
}
```

Both the approaches described above are part of the pattern Fire and Forget i.e We create a Thread and run some operation but don't wait for response.(caller doesn't need to know when or how they finish.) But what if we actually want a response? like ETAThread which returns ETA. Here comes the Callable Interface into play.

3) By implementing the Callable Interface

In Java, the Callable interface provides an enhanced way to implement multithreading when you need tasks to return a result. 

Since the run method from previous two approaches cannot return a response, we have a call() method in callable interface which needs to be overriden and configured with the datatype of result.


Now to create a thread you either need to extend the Thread class and do start or you pass the runnable instance to the Thread class and run start method. It doesn't directly accept the callable interface instance. So we use ExecutorService (that manages the thread) and Future interface.


The **submit()** method of ExecutorService returns a **Future** object, which we use to get the result of the task using the **get()** method.


```java 
// define the return type in class signature and method signature
class ETACalculationTask implements Callable<String>{
    // overide the call method
    @Override
    public String call() throws InterruptedException{
        Thread.sleep(3000);
        // businesss logic
        return "ETA is 25 mins"
    }
}

class Main{
    public static void main(String[] args){
        // this is a callable instance
        ETACalculationTask etaTask =new ETACalculationTask();

        // We use executorService to manage all threads and use submit method which returns a Future object for each of the thread. We can now perform get method on this Future object to get the result
        Future<String> etaResult = executorService.submit(etaTask);
        try{
            System.out.println(etaResult.get());
        }catch(InterruptedException e){
            e.printStackTrace();
        }

        
    }
}


```

The Future interface has couple of methods liek get() that blocks the main thread until the task completes and returns the result. Methods like isDone() and isCancelled() help in checking if task is completed or canceled.

The above example works seamlessly with the executorService, but what if we want to use callable object and create Thread directly, for that we use FutureTask Class (which implements Future and Runnable)


```java
class ETATask implements Callable<String>{
    @Override
    public String call(){
        // logic
        return "Eta is 25 mins";
    }
}

class Main{
    public static void main(String[] args){
        // callable instance is passed into FutureTask and then this object is passed to te Thread
        
        FutureTask<String> etaTask = new FutureTask<>(new ETATask());
        Thread etaThread = new Thread(etsTask);
        etaThread.start();
    }
}



```


#### Managing threads, Thread Pools, Executors Framework

Until now we were managing threads manually, which is risky, we dont close the threads, the process is error-prone and also very inefficient.


Creating a new thread for each task can quickly lead to an excessive number of threads, overwhelming the system and degrading performance. It can also lead to memory exhaustion (each thread consumes memory) and performance degradation (if thread fails to terminate post work).


Context Switching Overhead: Managing too many threads increases context switching, where the system saves and loads thread states. This overhead reduces overall system efficiency as the CPU spends more time switching between threads than doing real computational work.


Here comes into play, Thread Pools, where a fixed number of threads are available to handle the tasks.


**Executor Framwork** in Java is part of java.util.concurrent package.
It consists of Exceutor Interface(one method execute(Runnable)), ExecutorService Interface(extends Executor and has methods like submit(Runnable/Callable), shutdown(), shutdownNow()), ThreadPoolExecutor Class (concrete implementation of ExecutorService ) and Executors Class (utility class to create predefined types of executor services).



The difference between execute() and submit() method is execute() is used to run runnable tasks (which doesnot return anything), while submit(Runnable/Callable) is used when we want a response back. It returns a Future object on which we can call upon get method to get the result/response

```java

import java.util.concurrent.*;

// Future and Submit example
class FutureExample {
    public static void main(String[] args) throws Exception {
        ExecutorService executor = Executors.newFixedThreadPool(2);

        Future<Integer> future = executor.submit(() -> {
            Thread.sleep(1000);
            return 77;
        });

        System.out.println("Doing other work...");

        Integer result = future.get(); // blocks until result is ready
        System.out.println("Result: " + result);

        executor.shutdown();
    }
}
```

shutdown()
    Purpose: Initiates an orderly shutdown of the executor service. Once this method is called, the executor will stop accepting new tasks but will continue to process the tasks that have already been submitted.

shutdownNow():
    Purpose: Attempts to stop all actively executing tasks, halts the processing of waiting tasks, and returns a list of the tasks that were waiting to be executed.


Thread starvation occurs when a thread is perpetually unable to access the resources it needs for execution due to high contention, often caused by prioritizing certain threads over others. As a result, low-priority threads or threads with resource dependencies may never get a chance to execute.
 

Fairness ensures that all threads get an opportunity to execute, preventing some threads from being permanently blocked. A fair scheduler allocates CPU time evenly across all threads, preventing starvation.


3 Main Types of Thread Pools:

1) Fixed Thread Pool: It is used to create pool with a fixed number of threads. Once a task is submitted, the executor assigns it to an available thread from the pool. If all threads are busy, new tasks are queued until a thread becomes available.

```java
import java.util.concurrent.*;

// Future and Submit example
class FutureExample {
    public static void main(String[] args) throws Exception {
        ExecutorService executor = Executors.newFixedThreadPool(2);

        Future<Integer> future = executor.submit(() -> {
            Thread.sleep(1000);
            return 77;
        });

        System.out.println("Doing other work...");

        Integer result = future.get(); // blocks until result is ready
        System.out.println("Result: " + result);

        executor.shutdown();
    }
}
```

2) A Cached Thread Pool creates new threads as needed but reuses previously constructed threads when they are available. If a thread remains idle for more than 60 seconds, it is terminated and removed from the pool.


3) A Scheduled Thread Pool allows you to schedule tasks with fixed-rate or fixed-delay execution policies. It supports delayed or periodic execution of tasks, making it useful for scheduling tasks at regular intervals or after a specific delay. Let's understand with the code given:

```java
class SessionCleaner {
    public static void main(String[] args) {
        ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(1);

        Runnable task = () -> System.out.println("Cleaning up expired sessions...");

        scheduler.scheduleAtFixedRate(task, 0, 5, TimeUnit.SECONDS);
    }
}
```



#### Thread Safety

Thread safety means that a piece of code, object, or method works correctly and consistently when used by multiple threads at the same time. It makes sure that no wrong result is produced and no data gets corrupted — even if many threads are accessing or changing the same thing.


A simple real life example would be a counter, if two threads simultaneously inc the counter, there may be discrepancies.


When two or more tasks reach for the same data at the same moment, the first one to finish “wins,” and the final result depends on sheer timing — not on logic. That timing lottery is called a race condition.


Also counter++ is not a single operation, it consists of three tasks, reading the data, increasing it and finally writing it.


Ways to handle this race conditions:


1) Synchronized keyword:

We just saw how two threads can clash over the same data. To stop that clash, Java offers the synchronized keyword — a built-in way to let only one thread touch a critical section at a time.

When a thread enters a synchronized region, it grabs a monitor lock tied to an object (or the class itself).

If the lock is available: the thread enters and safely runs the code inside.
If the lock is already held by another thread: the thread waits (is blocked) until the lock is released.

This way, only one thread at a time can execute the synchronized part, which avoids conflicts and ensures thread safety.

```java
// Entire method is protected by the instance’s monitor lock
    public synchronized void increment() {
        count++;          // atomic under the lock
    }

    //alternative
    public void increment() {
        // Lock only the code that truly needs protection
        synchronized (lock) {
            count++;
        }
    }
```

2) volatile keyword:

The volatile keyword in Java is used to ensure visibility, not atomicity. It tells the JVM that a variable's value may be updated by multiple threads and that every read or write should go directly to and from main memory, rather than being cached in a thread’s local memory (CPU cache).


1. visibility
Any update made to a volatile variable by one thread becomes immediately visible to all other threads.
Without volatile: one thread might keep reading an old value from its local cache.
With volatile: all threads will always see the latest value in memory.


volatile is best suited for scenarios where:
One thread writes to a variable, and others only read it.
There’s no need for atomic operations, just fresh visibility.



3) Atomic variables

Java provides a set of classes under the java.util.concurrent.atomic package, designed to handle common types like integers and booleans in a thread-safe, high-performance way — without using locks.

All atomic classes use a technique called Compare-And-Swap (CAS) at the hardware level. This is what makes them lock-free and highly performant.

CAS stands for Compare-And-Swap. It is a low-level CPU/hardware instruction that checks if a memory location holds an expected value, and if so, swaps it with a new value — all in one atomic step. Here's how it works in simple terms:

“If the current value is what I expect it to be, update it with a new value. Otherwise, try again.”

Common Atomic Class:
AtomicInteger – for atomic operations on integers
AtomicBoolean – for managing flags safely


```java
import java.util.concurrent.atomic.AtomicInteger;

class PurchaseAtomicCounter {

    // A thread-safe integer backed by hardware-level CAS
    private final AtomicInteger likes = new AtomicInteger(0);

    // Atomically add 1 to the like counter
    public void incrementLikes() {
        int prev, next;
        do {
            // Step 1  – read the current value.
            //           (May be outdated if another thread wins the race.)
            prev = likes.get();

            // Step 2  – compute the desired next value.
            next = prev + 1;

            // Step 3  – attempt to swap: 
            /*          “If current value is still ‘prev’, set it to ‘next’.”
             *          Returns true on success, false if another thread
             *          already changed the value (retry needed).
             */
        } while (!likes.compareAndSet(prev, next));
    }

    // Read-only accessor
    public int getCount() {
        return likes.get();
    }
}

```
Compare-And-Set is a Java-level method (like AtomicInteger.compareAndSet()) that uses the hardware CAS (compare-and-swap) under the hood to implement safe, lock-free updates.

This works best for Simple counters or flags, in complex case synchronized is the way to go

#### So is synchronized keyword enough??

So, if we look back synchronized keyword actually locks the block or method. So one thread locks it and all the other threads are blocked.

Let's consider the scenario of ticket/seat booking in bookMyShow. If a thread enters this sync block and then goes idle, no other threads can access the lock and are blocked indefinitely. 

Why not just rely on synchronized?

No timeout support: the lock waits forever.
No explicit control over acquiring/releasing the lock.
You can’t interrupt a thread stuck waiting for a lock.
No guarantee of fairness: Some threads may wait longer than others.

#### Lock vs Mutex

|Lock|Mutex|
|:-|:-|
|General term for mutual exclusion| owners associated with lock|
|not always enforced, locked, no guarentee if unlocked|only the thread that acquires the lock can release it|
|synchronized keyword to use it|Reentrant lock used (like mutex)|
|We can use another thread to unlock it programmatically|Only the thread that locked it, can unlock it|
|Like a washroom door, anyone can access it when available|Home door, if I am the owner, I can only lock and unlock it.|


#### Reentrant Lock

Reentrant lock is a constuct used in java to achieve mutex.
The key operations that we do using Reentrant lock are lock, trylock(time, unit), unlock, etc.

When we use lock, it sees if lock is available or not, based on that it returns True or False. However, if we want to wait some time until the lock becomes available, we can use trylock(5000,ms), here we say try to acquire lock if it is in use wait for (5000ms) 5sec, if it is still not available then return False.

```java
import java.util.concurrent.locks.ReentrantLock;

class TicketBooking {
    // Number of seats initially available
    private int availableSeats = 1;

    // Dedicated lock for this shared resource
    private final ReentrantLock lock = new ReentrantLock();

    // Public method to book a ticket
    public void bookTicket(String user) {
        System.out.println(user + " is trying to book...");

        // Acquire the lock – blocks until available
        lock.lock();
        try {
            System.out.println(user + " acquired lock.");

            // Critical section: check and update shared state
            if (availableSeats > 0) {
                System.out.println(user + " successfully booked the ticket.");
                availableSeats--;
            } else {
                System.out.println(user + " could not book the ticket. No seats left.");
            }
        } finally {
            // Always release the lock in a finally block
            System.out.println(user + " is releasing the lock.");
            lock.unlock();
        }
    }
}

class Main {
    public static void main(String[] args) {
        // Shared instance of TicketBooking
        TicketBooking bookingSystem = new TicketBooking();

        // Create two threads representing two users trying to book at the same time
        Thread user1 = new Thread(() -> bookingSystem.bookTicket("User 1"));
        Thread user2 = new Thread(() -> bookingSystem.bookTicket("User 2"));

        // Start both threads
        user1.start();
        user2.start();

        // Wait for both threads to finish
        try {
            user1.join();
            user2.join();
        } catch (InterruptedException e) {
            System.out.println("Thread interrupted: " + e.getMessage());
        }
    }
}

```


Ok the blocking problem seems to be solved, but what about the thread using the lock gooing idle scenario?

For this case, we use a ThreadPoolScheduledExecutor(1) of size 1 and we fix a certain interval after which if the lock is not released, we release it safely.


```java
import java.util.concurrent.*;
import java.util.concurrent.locks.ReentrantLock;

// ───────────────────────── ExpiringReentrantLock ───────────────────────── 

// Lock with a built-in “auto-release after N ms” timer
class ExpiringReentrantLock {
    // underlying mutual-exclusion lock
    private final ReentrantLock lock = new ReentrantLock();

    // single-thread scheduler to run the expiry task
    private final ScheduledExecutorService scheduler =
            Executors.newSingleThreadScheduledExecutor();

    // flag that tells the expiry task a timed lock is still active
    private volatile boolean isLocked = false;

    // Tries to acquire immediately; if successful, schedules auto-unlock
    public boolean tryLockWithExpiry(long timeoutMillis) {

        // attempt immediate acquisition
        boolean acquired = lock.tryLock();
        if (acquired) {
            // mark as held under the timer
            isLocked = true;

            // schedule unlock after timeout
            scheduler.schedule(() -> {
                if (lock.isHeldByCurrentThread() || isLocked) {
                    System.out.println("Auto-releasing lock after timeout.");
                    unlockSafely(); // delegate to common unlock logic
                }
            }, timeoutMillis, TimeUnit.MILLISECONDS);
        }
        return acquired;
    }

    // Releases the lock either by the owner thread or the timer
    public void unlockSafely() {
        if (lock.isHeldByCurrentThread() || isLocked) {
            isLocked = false; // reset timer flag

            // only the owner may actually call unlock()
            if (lock.isHeldByCurrentThread()) {
                lock.unlock();
                System.out.println("Lock released.");
            }
        }
    }

    // Graceful shutdown for the scheduler
    public void shutdown() {
        scheduler.shutdownNow();
    }
}

// ───────────────────────────── Driver code ──────────────────────────────

public class Main {
    public static void main(String[] args) {
        // shared expiring lock
        ExpiringReentrantLock expLock = new ExpiringReentrantLock();

        /* Idle user grabs the lock, then “goes missing” for 5 s
           (expiry timer is 3 s) */
        Thread idleUser = new Thread(() -> {
            if (expLock.tryLockWithExpiry(3000)) {
                try { Thread.sleep(5000); } // simulate long idle
                catch (InterruptedException ignored) {}
                expLock.unlockSafely(); // in case timer fired
            }
        }, "IdleUser");

        /* Active user starts after 1 s and keeps retrying every 1000 ms
           until it finally books the ticket once the timer frees the lock */
        Thread activeUser = new Thread(() -> {
            try { Thread.sleep(1000); } catch (InterruptedException ignored) {}
            while (true) {
                if (expLock.tryLockWithExpiry(3000)) {
                    System.out.println("Active user booked!");
                    expLock.unlockSafely();
                    break;
                } else {
                    System.out.println("Active user still waiting...");
                    try { Thread.sleep(1000); } catch (InterruptedException ignored) {}
                }
            }
        }, "ActiveUser");

        // start threads
        idleUser.start();
        activeUser.start();

        // wait for both to finish
        try {
            idleUser.join();
            activeUser.join();
        } catch (InterruptedException ignored) {}

        // shut down scheduler
        expLock.shutdown();
    }
}

```

#### ReadWriteLock

Another variation available in Java is ReadWrite locks. Whenever there is a write operation ongoing by a thread, all threads are blocked to read or write, however in the case of read, only write threads are blocked, other read threads are allowed.

ReadWriteLock solves this by giving us two independent locks:

- Read lock – many threads can hold it simultaneously as long as no thread is writing.
- Write lock – exclusive; once held, it blocks all other readers and writers.


```java

class StockPriceData{
    private double price =100.00;
    private final ReadWriteLock lock= new ReentrantReadWriteLock();

    public void changePrice(double newPrice){
        lock.writeLock().lock()
        try{
            System.out.println("Price changing to "+newPrice);
            price=newPrice
        }
        finally{
            lock.writeLock().unlock();
        }
       
    }

    public void readPrice(){
        lock.readLock().lock();
        try{
            System.out.printf("%s read price: %.2f%n", Thread.currentThread().getName(), price);
        }
        finally{
            lock.readLock().unlock();
        }
    }
}

```

#### Semaphores

What if we want a limited nos of concurrent access to shared resource. For instance we want atmost 4 users to access Netflix at a given time. For such scenarios we use Semaphores

A Semaphore in Java which is a concurrency tool that maintains a fixed number of permits—like tokens. A thread must acquire a permit to proceed and release it after completing its task. If no permits are available, it either waits or fails based on the method used. This helps limit the number of threads accessing a resource at once, making it ideal for enforcing device limits, rate limiting, or managing connection pools.


```java

class NetflixAccount{
    private final Semaphores deviceSlots;

    public NetflixAccount(int maxDevices){
        deviceSlots = new Semaphores(maxDevices);
    }


    public void login(String user){
        System.out.println(user + " trying to log in...");
        
        if (deviceslots.tryacquire()){
            System.out.println(user + " successfully logged in.");
            return true;
        } else {
            System.out.println(user + " denied login - too many devices.");
            return false;
        }

    }

    public void logout(){
        System.out.println(user + " logging out.");
        deviceslots.release();
    }
}

```

#### Deadlocks

A deadlock is a situation that arises in multithreaded or concurrent applications when two or more threads are permanently blocked, each waiting for the other to release a resource. In this state, none of the threads can proceed, and the system essentially freezes in that part of execution. Deadlocks are one of the most notorious problems in concurrent programming, often difficult to detect and debug.

