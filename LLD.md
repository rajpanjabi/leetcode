# LLD

LLD dives into the specifics of implementing individual features
It involves low level design like implementing classes, modules, design patterns, it doesn't concern hld like caching, load balancers, api gateways, etc...


It's a crucial phase in the software development lifecycle that focuses on the detailed design of individual components or modules of a system.


It involves specifying the internal structure, algorithms, and data structures that will be used to implement the system's functionality. It also acts as a bridge between high-level design and actual coding.

**Steps for LLD:**

- Requirements gathering (user activity, user flow from start (input) to end (output)), determine entities, actions and expected outputs
- Break down the requirements
- Laying Down Use Cases
- UML Diagrams
- Model Problems (Design patterns that can be used)
- Implement Code


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

####




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


#### Managing threads, Thread Pools

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