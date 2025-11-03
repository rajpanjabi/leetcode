# LeetCode Notes

## Quick Reference & Important Methods

### Number Operations
- **Get last digit**: `number % 10` (e.g., `47 % 10 = 7`)
- **Get last n digits**: `number % 10^n` (e.g., `65789 % 100 = 89`)
- **Remove last n digits**: `number // 10^n` (e.g., `12345 // 100 = 123`)

### Dictionary Operations
- **Get key with max value**: `max(d, key=d.get)`
- **Get key with min value**: `min(d, key=d.get)`

### List/Iterable Operations
- **Zip multiple iterables**: `zip(list1, list2, list3)` - iterates multiple lists together
- **Any function**: Returns `True` if at least one element is `True`
  - Example: `any(char in sett for char in word)`

---

## Sorting Methods

### Merge Sort

**Divide and Conquer Algorithm**

**Steps**:
- Main function splits the input array until we have minmum i.e size 1 arrays by dividing the input array into halves.
- Then we merge these arrays recursively using the merge function and finally return the merged sorted array.



```python
def sortArray(self, nums: List[int]) -> List[int]:
        # mergesort consists of mainly 2 parts
        # dividing into subarrays, a.k.a main algorithm
        def mergesort(nums):
            # recursively break down into smaller subarrays
            # and merge the results
            if len(nums)<=1:
                return nums
            mid=len(nums)//2
            left=mergesort(nums[:mid])
            right=mergesort(nums[mid:])
            return merge(left,right)

        # merging algo
        def merge(arr1,arr2):
            i,j=0,0
            temp=[]
            while i<len(arr1) and j<len(arr2):
                if arr1[i]<=arr2[j]:
                    temp.append(arr1[i])
                    i+=1
                else:
                    temp.append(arr2[j])
                    j+=1
        
            temp.extend(arr1[i:])
    
            temp.extend(arr2[j:])
            return temp
        return mergesort(nums)
```


## Arrays

### Sorting in Python

#### Two Methods Available:

**1. `list.sort()` - In-place sorting**
- Sorts the list in place
- Returns `None`, so use: `list.sort()` then `print(list)`
- Only defined for lists
- More efficient if you don't need the original list

**2. `sorted(list)` - Creates new sorted list**
- Creates a new sorted list
- Returns the sorted list: `print(sorted(list))`
- Original list remains unchanged
- Accepts any iterable

#### Sorting Parameters:

**Key Parameter**: Specify a function to be called on each element before comparison
```python
student_tuples = [
    ('john', 'A', 15),
    ('jane', 'B', 12),
    ('dave', 'B', 10),
]
sorted(student_tuples, key=lambda student: student[2])   # sort by age
# Output: [('dave', 'B', 10), ('jane', 'B', 12), ('john', 'A', 15)]
```

**Reverse Parameter**: For descending order
```python
list.sort(reverse=True)
```

### Longest Subarray with Sum k

This problem has 2 variations , onw ith just the positive numbers, other with positive,zeroes and negative numbers.
There are 3 approaches to dolve this problem brute force, better and optimal

#### Brute Force

```python
    def longestSubarraySumK(int[] nums, int K):
        # the easiest and naive approach to this problem would be try all the possible subarrays and then check if the subarray sum equals to k, also maintain the longest (maxlength subarray)
        longest=0
        for i in range(len(nums)):
            currSum=0
            for j in range(len(nums)):
                currSum+=nums[j]
                if currSum==k:
                    longest=max(longest,j-i+1)
        return longest  
```
This solution takes O(n^2) time, which is inefficient.

#### Better Solution
Here we can try to use some math logic if a subarray sum equals to x and we have already seen a subarray with sum eqaul to x-k, then the remaining subarray definitely adds up to k
[............]
[-(x-k)-][-k-]
[-----x------]
In this we maintain a regular or prefix sum variable, now if prefixsum eqauls to k, we store the length directly since we start from 0 and the sum eqausl to k, the lenght will be index+1(0 based, so+1). In other case we might have a subarray (with sum ==k) inside this array that might actually end at the index we are standing (current index), to check that we look for subarray (x-k). so if a (x-k) subarray exists and we are with sum =x, then the remaining subarray would be summing up to k. For this at each iteration, we store the prefixsum and its index in a hashmap

```python
    def longestSubarraySumK(int[] nums, int K):
        hashmap={}
        prefixSum=0
        longest=0
        for i in range(len(nums)):
            prefixSum+=nums[i]
            if prefixSum==k:
                longest=max(longest,i+1)
            # in other case
            if prefixSum-k in hashmap:
                longest=max(longest,i-hashmap[prefixSum-k])
            # we store sum ending at each index for future lookups
            # another edge case here is to store prefixsum with index only if its not in the hashmap because of 0's
            # [3,0,0,0,1,1] here prefixsum=3 for indices 0,1,2,3. so if we keep on updating it would only store sum=2 for index 3. the subarrays with sum=3 are [3],[3,0],[3,0,0,0]  
            # when we reach prefixsum=5 and try to look for x-k (k=2) 5-2=3 3 exists in hashmap 3->3(iidx)so length would be 5-3=2 but actually longest should be 5-0 [0,0,0,1,1]
            if prefixSum not in hashmap:
                hashmap[prefixSum]=i
        return longest
```
This approach uses extra space i.e hashmap O(n) because for each index we store sum-> index
The Time Complexity is O(N)

#### Optimal Solution
The most optimal solution is using 2 pointers

```python
    def longestSubarraySumK(int[] nums, int K):
        l=0
        r=0
        longest=0
        currSum=nums[0]
        while r <len(nums):
            while left<right and currSum>k:
                currSum-=nums[l]
                l+=1
            if currSum==k:
                longest=max(longest,r-l+1)
            if r<len(nums):
                r+=1
                currSum+=nums[r]    
            
        return longest    

```

### Matrix Zeroes
Given an m x n integer matrix matrix, if an element is 0, set its entire row and column to 0's.
You must do it in place.

Now the brute force would be to use some extra space probably two extra arrays (one for row, one for col) or use two sets.
Now traverse the matrix and if matrix[i][j]==0 then add that i, j to the sets.
Now we traverse all the rows present in rows set and set all the values equal to zero for that row 
Similarly for all the cols present in cols set, we set all values equal to zero
This would be O(mxn) time soltuion with O(m + n) extra space

```python
    class Solution:
        def setZeroes(self,matrix: List[List[int]])->None:
            rows0=set()
            cols0=set()
            rows,cols=len(matrix),len(matrix[0])
            for i in range(rows):
                for j in range(cols):
                    if matrix[i][j]==0:
                        rows0.add(i)
                        cols0.add(j)
            for row in rows0:
                for j in range(cols):
                    matrix[row][j]=0
            for col in cols0:
                for i in range(rows):
                    matrix[i][col]=0
```

Better solution would be to do in place. 
If we carefully observe, if a row has a zero then all the elements in that row would be set to zero, similarly for cols.
SO we use the first row and first cols to mark if the row or col needs to be zeored, else we keep the original values as is.
Another imp thing here is to mark if the first row and col has any 0, in that case we also need to make them zeroes at the end, so we use two boolean markers for that purpose
This would also take O(mxn) time , but only use O(1) space

```python
class Solution:
    def setZeroes(self,matrix: List[List[int]])->None:
        rows,cols=len(matrix),len(matrix[0])
        # we use two boolean flags
        first_row_zero, first_col_zero = False, False
        for j in range(cols):
            if matrix[0][j]==0:
                first_row_zero = True
                break
        for i in range(rows):
            if matrix[i][0]==0:
                first_col_zero=True
                break
        # Now we check the entire matrix from 1st row and 1 st col and mark 0th row and 0th col as 0 if we find any zeroes
        for i in range(1,rows):
            for j in range(1,cols):
                if matrix[i][j]==0:
                    matrix[i][0]=0
                    matrix[0][j]=0
        # Now we know which rows and cols needs to be zeroed, we traverse final time to mark them zero
        for i in range(1,rows):
            for j in range(1,cols):
                if mat[i][0]==0 or mat[0][j]==0:
                    mat[i][j]=0
        # Now the last task is to convert the first row and first col to zero if boolean says so
        if first_row_zero:
            for j in range(cols):
                mat[0][j]=0
        if first_col_zero:
            for i in range(rows):
                mat[i][0]=0

```

### Rotate Image
You are given an n x n 2D matrix representing an image, rotate the image by 90 degrees (clockwise).
You have to rotate the image in-place

Brute Force:
Intuition:
We use an answer matriux and store the ans. So we if carefully observe, we see a pattern between i and j i.e 
elt at mat[i][j] is moved to mat[j][n-1-i]
So we traverse the matrix and store the values from matrix[i][j] to answer [j][n-1-i]
This approach takes O(n^2) time and O(n^2) space.

```python
class Solution:
    def rotate(self, matrix:List[List[int]])->List[List[int]]:
        n = len(matrix)
        rotated = [[0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                rotated[j][n - i - 1] = matrix[i][j]
        return rotated
```

Optimal Solution:
Intuition:
Here we swap the elements above the diagonal with those below it. Now if we reverse the rows, we get the desired output.
Remember we traverse i from 0 to n, and j from i+1 to n
```python
class Solution:
    def rotate(self, matrix:List[List[int]])->List[List[int]]:
        n=len(matrix)
        # swap elements on the sides of diagonal
        for i in range(n):
            for j in range(i+1,n):
                if i!=j:
                    matrix[i][j],matrix[j][i]=matrix[j][i],matrix[i][j]
        # now swap the elments beside the middle line 
        #  reverse the entire matrix
        for i in range(n):
            matrix[i].reverse()
```

### Spiral Matrix

Given an m x n matrix, return all elements of the matrix in spiral order.

Solution:
If you just think about the answer and how can you traverse the matrix and get to it, you'll see a pattern. Just recurse on hte apttern until 
you run out of boundary. 

```python
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        rows,cols=len(matrix),len(matrix[0])
        res=[]
        def traverse(i,j,ei,ej):
            # base case or stopping condition
            if i>=ei or j>=ej:
                return
            # add elements from first row
            # row would be mat[i][0] to mat[i][cols-1-i]
            for k in range(j,ej):
                res.append(matrix[i][k])

            # add elements from last cols
            # col would be 
            for k in range(i+1,ei):
                res.append(matrix[k][ej-1])

            # last row in reverse order
            # travel from last col-1 (since we added it already in prev) to jth col
            if ei - 1 != i:

                for k in range(ej-2, j-1,-1):
                    res.append(matrix[ei-1][k])
            # remaining from first col reverse from last row to first row
            if ej - 1 != j:
                for k in range(ei-2,i,-1):
                    res.append(matrix[k][j])
            # now we update the variables we used
            # now we call again with these updated values 
            # switch by one col and repeat
            traverse(i+1,j+1,ei-1,ej-1)
        traverse(0,0,rows,cols)
        return res 

```

### Subarray sum equals k
Given an array of integers nums and an integer k, return the total number of subarrays whose sum equals to k.
A subarray is a contiguous non-empty sequence of elements within an array.

Brute Force:
Idea would be to check all the subarrays, and then count only those whose sum adds up to k.
Since we are trying all possibilities of subarray, this will take two nested loops, resulting in O(n^2) solution.

```python

class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        # brute force, try all subarrays and return those whose sum ends up as k
        subarrays=0
        n=len(nums)
        for i in range(n):
            currsum=0
            for j in range(i,n):
                currsum+=nums[j]
                if currsum==k:
                    subarrays+=1
        # return subarrays

```

Optimised:
To optimise this, we might need some extra space and try to reduce the time complexity. What are we trying to do here is check if subarray sums eqaul to k. We can use a math trick, so for indices i,j s.t  `i<j`. Now `sum(0->i)`  and `sum(0->j)`. there might be a window of i-j since `i<j`. So sum of this window can be broken as `sum(0->j)-sum(0->i)`. If we somehow know if we have seen some sum and then for every value of currsum check that, we can get hold of those subarrays.
We use a hashmap and store runningsum at each index. Now the key would be the runningSum and the value would be the freq/nos of times we have seen that sum, we do this because we need all the subarrays.

```python   
class Solution:
    def subarraySum(self, nums: List[int], k:int) -> int:
        hmap=defaultdict()
        cnt=0
        runningSum=0
        for i in range(len(nums)):
            runningSum+=nums[i]
            # Now we look for runningSum-k in the hmap
            if runningSum-k in hmap:
                cnt+=hmap[runningSum-k]
            hmap[runningSum]+=1
        return cnt

```








**There are 3 Important array algorithms**
-**Kadane's Algorithm**

In Brute Force, I try all the possible subarrays, I maintian a running sum for subarray starting from each index(i) and ending at variable indices(j) and fianlly check if the maxSum needs to be updated

For Kadane's algorithm which is the optimal version for this problem, the base idea is relatively simple, You iterate once trhought ht array and you maintian two variables ,one is the global maxSUm and one is the running sum. At each iteration, you add the curr elt's val to the runningsum , now you update the maxsum if required , finally you update the currsum =0 incase if it goes below zero. We do this because if the currsum goes below zero, it wouldnt' help us in finding maxsum anymore, it will rather diminish our returns in future sum too, so its better to start fresh

```python
    def maxSubArray(self, nums: List[int]) -> int:
            # Brute force
            # maxSum=float('-inf')
            # for i in range(len(nums)):
            #     currSum=0
            #     for j in range(i,len(nums)):
            #         currSum+=nums[j]
            #         maxSum=max(maxSum,currSum)
            # return maxSum

            # Optimal approach
            currsum=0
            maxsum=float('-inf')
            for i in range(len(nums)):
                currsum+=nums[i]
                maxsum=max(currsum,maxsum)
                if currsum<0:
                    currsum=0
            return maxsum

```



-**Moore's Voting Algorithm - Majority Element in an array (>n/2)elts**
**Problem** Given an array `[2,2,1,1,1,2,2]` with one element whose freq is greater then n/2, return that elt

```python
    def majorityElement(nums: List[int]) -> int:
        cnt=0
        elt=0
        for i in range(len(nums)):
            # check if cnt==0
            if cnt==0:
                cnt=1
                elt=nums[i]
            elif nums[i]==elt:
                cnt+=1
            else:
                cnt-=1
        return elt
```
Why thi works? 


-**Dutch Flag Algorithm**




---

## Stack

### Next Greater Element (NGE)

**Problem**: Given array `[6,0,8,1,3]`, return array with NGE of each element or `-1`
- Output: `[8,8,-1,3,-1]`

**Brute Force**: O(n²)
- For each element at index `i`, loop through `j = i+1` to `n` to find NGE

**Optimized Solution**: O(n) time, O(n) space
- Start from the rightmost element (no NGE = `-1`)
- Use a stack to store elements
- Iterate backwards:
  - If stack top > current element → NGE found
  - Else → pop from stack until empty or NGE found
  - Push current element to stack

**Why it works**: Think of lamp posts `[5,3,2,6,7]` - elements 3,2 won't be visible from 5, only 6 will be

**Similar**: Previous Smaller Element (opposite logic using stack)

### NGE II (Circular Array)

**Problem**: Find NGE in circular array

**Brute Force**: O(n²)
- Treat array as doubled: `[2,10,12,1,11]` → `[2,10,12,1,11,2,10,12,1,11]`
- For `i = 0 to n-1`, check `j = i+1 to i+n-1`
- Use modulo for circular indexing: `j % n`

**Optimized Solution**: O(n) using stack
```python
def nextGreaterElements(self, nums: List[int]) -> List[int]:
    stack = []
    n = len(nums)
    nge = [-1] * len(nums)
    for i in range(2*(n-1), -1, -1):
        i = i % n
        while stack and stack[-1] <= nums[i % n]:
            stack.pop()
        if stack and i <= n-1:
            nge[i % n] = stack[-1]
        stack.append(nums[i % n])
    return nge
```

### Largest Rectangle in Histogram

**Problem**: Find largest area rectangle in histogram `heights = [2,1,5,6,2,3]`

**Approach**:
- For each bar, find how far it can expand left and right
- Find **Next Smaller Element (NSE)** on right
- Find **Previous Smaller Element (PSE)** on left
- Rectangle can only extend while height ≥ current bar height

**Solution**:
1. Precompute NSE and PSE using stack
2. For each index: `area = heights[i] * (nse - pse - 1)`
3. Return maximum area

**Further Optimization**: Calculate PSE and NSE on the fly while computing area

---

## Linked List

### Structure
- Linear data structure (chain of nodes)
- Each node contains: **data** and **next** pointer
- Elements not stored contiguously (unlike arrays)
- Best for variable size data

### Important Tricks

#### Things to do for a linked list questions:
-Always account or check if single node or no node present
-results vary for odd and even lengths, account for that as well



#### 1. Deletion Without Head (LC 237)
[Problem Link](https://leetcode.com/problems/delete-node-in-a-linked-list/)

**Given**: Node to delete (not head)
**Example**: `2->3->5->9->11`, delete node `5`

**Solution**:
```python
node.val = node.next.val          # Copy next value: 2->3->9->9->11
node.next = node.next.next        # Bypass duplicate: 2->3->9->11
```

#### 2. Tortoise and Hare Algorithm (Slow-Fast Pointers)

**Uses**:
- Find middle of linked list
- Detect cycle in linked list

**Finding Start of Loop**:
- **Naive**: Use hashmap to track visited nodes → O(n) space
- **Optimized**:
  1. Find collision point using slow/fast pointers
  2. Reset slow pointer to head
  3. Move both pointers by 1 unit
  4. New collision point = start of loop

#### 3. Odd Even Linked list

given head of a linked list
if single or no node, return head
in other cases, odd=head, even=head.next, evenHead=even
now while we have even and even.next:
odd.next=even.next; odd=odd.next
even.next=odd.next; even=even.next

finally connect odd.next=evenhead
return evenhead



---

## Dynamic Programming

### Count Square Submatrices (LC 1277)
[Problem Link](https://leetcode.com/problems/count-square-submatrices-with-all-ones/)

**Problem**: Count all square submatrices with all ones

**Brute Force**:
- For each index, check all possible squares with it as top-left
- Count 1x1, 2x2, 3x3... squares
- Sum up all squares

**DP Approach**: O(m×n)

**Idea**: `dp[i][j]` = number of squares with `[i][j]` as bottom-right cell

**Base Cases** (1st row and 1st column):
```python
for i in range(m):
    if matrix[i][0] == 1:
        dp[i][0] = 1

for j in range(n):
    if matrix[0][j] == 1:
        dp[0][j] = 1
```

**DP Transition**:
```python
for i in range(1, m):
    for j in range(1, n):
        if matrix[i][j] == 1:
            dp[i][j] = 1 + min(dp[i-1][j], dp[i-1][j-1], dp[i][j-1])
```

**Final Answer**: Sum all values in `dp` matrix
```python
squares = sum(sum(row) for row in dp)
```

---


## Binary Search

If the problem is about searching and the given array is sorted, then we always go for Binary Search. The cored idea behind binary search is to eliminate the search space by (1/2) in each iteration. n, n/2, n/4, n/8...1
if the final element ends up as target, return it else return -1

``` python
    def binarySearch(nums:List[int], target: int) -> int:
        n=len(nums)
        l=0
        r=n-1
        while l<=r:
            mid=(l+r)//2
            if nums[mid]==target:
                return mid
            elif nums[mid]>target:
                r=mid-1
            else:
                l=mid+1
        return -1
```

### Search in Rotated Sorted Array

Problem Statement: Given a sorted array which may be left rotated at some index k such that elements before k are sorted and after k are sorted.
[0,1,2,4,5,6,7] k=3 => [4,5,6,7,0,1,2]

Here we are supposed to find the solution in O(logn)
If we try to do linear search , it will take O(n), if we try to find the rotation point that too will end up taking O(n).
So instead focus on what binary search offers, which is eliminating the search space into half
Now the problem is how to figure out if we go left or right?
If we observe carefully, we can clearly see that either left side or right side is sorted.
Now binary search only works on a sorted array, so we check which portion is sorted, after that we check if target lies within the let side or not , if yes we do binary search on left siude, else we move to the right side


```python
    def search(nums: List[int], target: int) -> int:
            n=len(nums)
            l,r=0,n-1
            while l<=r:
                mid=(l+r)//2
                if nums[mid]==target:
                    return mid
                elif nums[l]<=nums[mid]:
                    # left is sorted 
                    if nums[l]<=target<=nums[mid]:
                        r=mid-1
                    else:
                        l=mid+1
                else:
                    if nums[mid]<=target<=nums[r]:
                        l=mid+1
                    else:
                        r=mid-1
            return -1


```

### Search in Rotated Sorted Array 2

Now in this problem, we have the array that might be rotated and also has duplicates, in our previous solution we applied regular binary search logic, and then on the top of that to divide the search space into half we first checked for the sorted portion, then checked if the target existed in hte rnage and moved pointers accordingly.
Well this approach works just fine for most of the cases, but when low==mid==high, in this case we are not able to figure out which portion (left or right) is sorted, so in this case we simply do l+=1, r-=1 and continue to next iteration. 

```python
    def binarySearchRotated2(nums: List[int], target: int)-> int:
        n=len(nums)
        l,r=0,n-1
        while l<=r:
            mid=(l+r)//2
            if nums[mid]==target:
                return True
            if nums[mid]==nums[l]==nums[r]:
                l+=1
                r-=1
                continue
            if nums[l]<=nums[mid]:
                if nums[l]<=target<=nums[mid]:
                    r=mid-1
                else:
                    l=mid+1
            else:
                if nums[mid]<=target<=nums[r]:
                    l=mid+1
                else:
                    r=mid-1
        return False

```

### Search Min in Rotated Sorted Array

```python
    def findMin(self, nums: List[int]) -> int:
        # sorted
        # no dups
        # rotated
        # return the min element of this array using binary search O(logn)
        n=len(nums)
        l,r=0,n-1
        minimum=float('inf')
        while l<=r:
            mid=(l+r)//2
            # check if left half is sorted
            if nums[l]<=nums[mid]:
                minimum=min(minimum,nums[l])
                l=mid+1
            else:
                minimum=min(minimum,nums[mid])
                r=mid-1
        return minimum
        

```

### Single element in sorted array

```python
class Solution:
    def singleNonDuplicate(self, nums: List[int]) -> int:
        n=len(nums)
        if n==1:
            return nums[0]
        l,r=1,n-2
        if nums[l]!=nums[l-1]:
            return nums[0]
        if nums[n-2]!=nums[n-1]:
            return nums[n-1] 

        while l<=r:
            mid=(l+r)//2
            if nums[mid-1]!=nums[mid] and nums[mid]!=nums[mid+1]:
                return nums[mid]
            # left or right?
            # idx is even or odd
            if (mid%2==0 and nums[mid]==nums[mid-1]) or (mid%2==1 and nums[mid]==nums[mid+1]):
                r=mid-1
            else:
                l=mid+1
        

```











## Python Utilities

### Zip Function
Iterate multiple lists together:
```python
languages = ['Java', 'Python', 'JavaScript', 'Go']
versions = [14, 3, 6, 8]
years = [2002, 2007, 2011, 2014]

for lang, ver, yr in zip(languages, versions, years):
    print(lang, ver, yr)
```

**Output**:
```
Java 14 2002
Python 3 2007
JavaScript 6 2011
Go 8 2014
```

### Bits 
The numbers that we human use like 0,1,2,3,4,5,6,7,8,9 (10 numbers in total) are in decimal format. i.e base 10
The numbers that the computers use 0 and 1 (2 nos in total) are in binary format i.e base 2

To convert the decimal to binary we divide the given decimal number by 2 until we reach one as remainder,. Now we take all the remainders in reverse order to form the binary representation of the decimal nos

binary of 7 would be 7/2= (2*3) 1(remainder) ==3/2(2*1) 1 (remainder). Now reverse 111=7

1's complement
lets say for 13 we want to find 1's complement
13 in its bianry form is (1101). now we flip all th bits-> (0010). this would be 1's complement

for 2's complement we add 1 to 1's complement of a number, so for 13 it would be 0010+1(0001)=0011

Operations in bits
AND & if both 1's then 1 else 0

Or | if one 1 then 1 , if both 0 then 0

XOR ^ if nos of ones=odd =1
if nos of ones=even =0

Any nos's XOR with 0=number itself
3^0=3
4^0=4


>> Right Shift. This pushes the numbers in binary format off the clif, shifts them by k positon to the right
13>>1 13(1101) 13>>1==(0110)(6)
13>>2 (1101)>>2 (0011) 3

so number>>k == number//2^k

Now how about negative numbers. the binary format uses the first bit (leftest side)(31st bit for int, since int has 32 bits starting from 0 to 31) to store the sign, so if last bit is 0, number is positive, however if it is one(1) then the number is negative.
negative numbers store in the positive number's 2's complement form
How is it stored
To store -13 in binary, the computer first calculates binary of 13 (00000...1101). Now it finds 2's complement by flipping bits and adding ones so it would be 11111....0011
What's the maximum size number that cna be stored so, all bits except the last bit with 1's would be the largest number so 01111..111 (32 bits)

Left shift is opposite to right shift push numbers to the right by 1 place so increase in number by 2
if 13== 13*2==26

Not operation
int x=5;
if x=~5
then first step is to flip
Now if negative number then store 2's complement
if not stop and jsut store the flipped bits


Rotated Array Notes

[1,2,3,4,5,6,7]
k=3
if I were to rotate an array by k places to the right, it would become
[5,6,7,1,2,3,4]

The simple or brute force solution here would be copy the last k places in a temp array temp=[5,6,7]
Now the remaining array can be modified by shifting them right . To achieve that, I would start from kth index and copy elements from i-k th
for i in range(k,n):
nums[i]=nums[i-k]

Now we again traverse from start of nums to position k and copy the elements from the temp
for i in range(k):
nums[i]=temp[i]

Optimal solution would be to reverse [0:d] and [d:n], and then finally reverse the entire array

