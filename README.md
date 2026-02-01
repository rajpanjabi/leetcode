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


<!-- A subtle change in above algorithm can improve space, instead of passing the array, we just pass start and end, so for the first time we pass (0,len(nums)-1) -->


```python

class Solution:
    def mergeSort(self, start, end):
        if start==end:
            # size of nums==1, safe to return
            return nums[start]
        # else we divide into halves
        mid=(start+end)//2
        left=mergeSort(start,mid)
        right=mergeSort(mid+1,end)
        return merge(left,right)

    
    def merge(nums1,nums2):
        # merge two arrays into one sorted array
        i,j=0,0
        res=[]
        while i<len(nums1) and j<len(nums2):
            if nums1[i]<=nums2[j]:
                res.append(nums1[i])
                i+=1
            else:
                res.append(nums2[j])
                j+=1
        # now either nums1 exist or nums2 
        while i<len(nums1):
            res.append(nums1[i])
            i+=1

        while j<len(nums2):
            res.append(nums2[j])
            j+=1
        return res


```

### Quick Sort

This is also a recursive algorithm, it takes O(nlogn) time and O(1) space. However, this algorithm is not stable, that is if duplicates in the array, the relative order may not hold true in final sorted array. Also time complexity heavily relies on how pivot is chose.

Steps:
Choose a pivot(usually the first or alst element)
Now take i and j pointers from start+1 and n-1 positions. Until they cross each other, swap pairs. the pairs would be element higher than pivot on left and element smaller then pivot on right. these values are swapped and we move i and j. Continue until i`<`j. Finally swap the pivot and the val at j the index, so pivot is at its right place



```python
class Solution:
    def QuickSort(nums:list[int])->list(int):
        n=len(arr)

        def partition(low:int, high:int)->int:
            pivot=nums[low]
            # lets take 1st elt as the pivot
            i,j=low-1,high+1
            while i<j:
                # find elt greater than that at pivot using i from start
                while nums[i]<=pivot and i<low-1 :
                    i+=1
                # find elt smaller than that at pivot using j from back
                while nums[j]>pivot and j>=low+1:
                    j-=1

                # if they are valid i<j then swap 
                if i<j:
                    nums[i],nums[j]=nums[j],nums[i]
            
            # once the loop ends, place the pivot at its correct place i.e where j stands (the last smaller elt than pivot )
            nums[low],nums[j]=nums[j],nums[low]




        # init starts with low=0 and high =len(nums)-1
        def qs(low,high):
            # if just one elt already sorted (low==high)
            if low<high:
                partitionIdx=partition(low,high)
                # recrusively sort the left and right halves
                left=qs(low,partionIdx-1)
                right=qs(partionIdx+1,high)
        
        qs(0,len(nums)-1)
        return nums




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

### Check if array is sorted

Given a array, that may be rotated, check if the original array was sorted

e.g[4,5,6,1,2,3] here the original array was [1,2,3,4,5,6] which on rotation by 3 places becomes [4,5,6,1,2,3]

Here the idea is to first find the pivot point, once found we reverse the left half, the right half and then the entire array. Now check if the array is sorted

```python

class Solution:
    def check(self, nums: List[int]) -> bool:
        # brute force try all k rotations
        # find k point of rotation
        # before point of rotation and after it sorted
        k=0
        n=len(nums)
        for i in range(1,len(nums)):
            if nums[i-1]>nums[i]:
                k=i
                break
        l,r=0,k-1
        while l<r:
            nums[l],nums[r]=nums[r],nums[l]
            l+=1
            r-=1
        l,r=k,n-1
        while l<r:
            nums[l],nums[r]=nums[r],nums[l]
            l+=1
            r-=1
        l,r=0,n-1
        while l<r:
            nums[l],nums[r]=nums[r],nums[l]
            l+=1
            r-=1
        for i in range(1,n):
            if nums[i-1]>nums[i]:
                return False
        return True
```


### Rotate Array

Naive way O(N) time and O(N) space

```python
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        # Brute Force O(N) Time and O(N) space solution
        # n=len(nums)
        # if k==0 or k==n:
        #     return
        # if k>n:
        #     k=k%n
        # res=nums[n-k:]
        # res+=nums[0:n-k]
        # for i in range(n):
        #     nums[i]=res[i]

        # Optimal Solution O(N) and O(1) Space
        n=len(nums)
        if k==0 or k==n:
            return
        k=k%n
        # now reverse both the parts (left of pivot and right of pivot)
        l,r=0,n-k-1
        while l<r:
            nums[l],nums[r]=nums[r],nums[l]
            l+=1
            r-=1

        l,r=n-k,n-1
        while l<r:
            nums[l],nums[r]=nums[r],nums[l]
            l+=1
            r-=1

        # finally reverse the whole array
        l,r=0,n-1
        while l<r:
            nums[l],nums[r]=nums[r],nums[l]
            l+=1
            r-=1
```


### 128. Longest Consecutive Sequence

Given an unsorted array of integers nums, return the length of the longest consecutive elements sequence.
You must write an algorithm that runs in O(n) time.

Intuition: Since we need to do it in o(n), definitely cannot sort the array.
<!-- length of longest consecutive elements, if i try to see how much each elt can elongate, and record the max i cna do it, but the array is not sorted, how can i do it in O(n). Lets see if I could use some extra space and do in linear time. firstly I want to only check how long an elt can go if its a starting point. for example in array [8,4,6,5,13,15,14], here 4 is  a starting point as well as 13. the other elts are the followers, so to find the starting point i need to check if their previous elts exists or not, dso i chekc for each elt if prev of 3 exists? prev of 4 exists? prev of 5 exists? and so on.. but if I do this i have to do linear search and this would go to O(n^2). But if i use extra space lets say i add all elts to a set initially . Now if I check for each elt it would take O(1). nice right. So add all elts to a set intially. O(n) time . Now iterate over the array and check if prev elt exists or not? if not this is a start point, conrtinue from this elt and check until its next elt exists, so for 4 , 3 doesnt exist, which makes it as a good start point, now inc4 by 1 that is 5, chekc if 5 exists? if yes, check for 6 and so on.. until next elt exists. finally record lenght and upate max if required. now continue for rest of elts to see if they are starting point, if yes, do the same ofr them and finally return the longest lenght discovered
-->
```python
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        sett=set(nums)
        n=len(nums)
        longest=0
    
        for num in sett:
            if num-1 not in sett:
                j=num
                while j in sett:
                    j+=1
                longest=max(longest,j-num)
        return longest
```


### Next Permutation

Given an array of integers nums, find the next permutation of nums.

Intuition:
For brute force if we find all permuations, and add it to an array , it takes n! to generate all permutations. to copy the list of permuations(of length n i.e same size of original array and adding it to res array) would take O(n). for n! permautaitons, it would be O(n!)x O(n). then we can find the next permuation by doing linear search. very inefficient

For better optimal version, we try to find next word. the next word should have the longest common prefix in starting so that it indeed is the next permutation. So we start from backwrds and try to find the dip point which is the poitn whose val is greater than the next index. this is the palce that if swapped can make the number actually greater. 
If we dont find a dip that is the array is in decreasing order liek 5,4,3,2,1, so the next permuation is reverse of nums.
However, if we do find a diop point, we then find again from back of array the element greater than the dip point and swap it. 
Now we reverse the elts from dip point +1 to the end of array (as it was increasing from backwards), thsi would sort the numbers and give us the smallest next permutation possible for a number.


```python
class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        # next permutation of a number is apparently nothing but the number bigger than the curr one by just min change. the idea is to replace the number at unit place by the first number less than it starting from left to right 
        # find the dip point
        # start from end
        n=len(nums)
        breakpoint=-1
        for i in range(n-2,-1,-1):
            if nums[i]<nums[i+1]:
                breakpoint=i
                break
        # now if we traverse the entire array and didn't find a breakpoint that means it is strcitly incresaing from back side 5,4,3,2,1, so the next poermuation is bascially the first ie. reverse of nums
        if breakpoint==-1:
            return nums.reverse()
        # if breakpoint found, i need to swap that index with someone greater from the idx+1 to len range, such that it is minimum i that rnage and greater than our elt at breakpoint
        for i in range(n-1,-1,-1):
            if nums[i]>nums[breakpoint]:
                nums[i],nums[breakpoint]=nums[breakpoint],nums[i]
                break
        # now finally reverse the elts after breakpoint, since they were increasing from end , if we reverse it will become decreasing to get the smallest possible next permuatation
        l,r=breakpoint+1,n-1
        while l<r:
            nums[l],nums[r]=nums[r],nums[l]
            l+=1
            r-=1
        
       
```



### Longest Subarray with Sum k

This problem has 2 variations , onw ith just the positive numbers, other with positive,zeroes and negative numbers.
There are 3 approaches to solve this problem brute force, better and optimal

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
In this we maintain a regular or prefix sum variable, now if prefixsum equals to k, we store the length directly since we start from 0 and the sum eqauls to k, the lenght will be index+1(0 based, so+1). In other case we might have a subarray (with sum ==k) inside this array that might actually end at the index we are standing (current index), to check that we look for subarray (x-k). so if a (x-k) subarray exists and we are with sum =x, then the remaining subarray would be summing up to k. For this at each iteration, we store the prefixsum and its index in a hashmap

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


### Contiguous Array

Brute Force:
Try all possible subarrays, check if zeroes and ones are equal, update max if required. O(n^2)

```python
class Solution:
    def findMaxLength(self, nums: List[int]) -> int:
        # Brute Force, try all possible subarrays, check if zeroes and ones are equal, update max if required. O(n^2)
        maxlength=0
        n=len(nums)
        for i in range(n):
            zero=0
            one=0
            for j in range(i,n):
                if nums[j]==0:
                    zero+=1
                else:
                    one+=1
                if zero==one:
                    maxlength=max(maxlength,2*(zero))
        return maxlength
```
Optimised Solution:

We use hashmap, the trick here is to convert 0 to -1, and 1 as 1, now if we calculate prefixsum, whenever there are equal nos of 0's and 1's the sum would be zero. Other thing to handle here is when the prefixsum at two indices are same then also the nos of 0's and 1's are same.

Why???

Let’s say you are at index i with a prefix sum = x. Now, at index j, your prefix sum is again x. This means that between i and j, the number of ones and zeros must be equal. For example, if you added three ones (prefix sum becomes x + 3) and later three zeros (prefix sum goes back to x + 3 - 3), you return to the same prefix sum. So, having the same prefix sum at two different indices indicates an equal number of 0s and 1s between them. Hope that makes sense.

```python
class Solution:
    def findMaxLength(self, nums: List[int]) -> int:
        prefixsum=0
        n=len(nums)
        longest=0
        seen={}

        for i in range(n):
            # prefixsum at index i
            if nums[i]==0:
                prefixsum-=1
            else:
                prefixsum+=1
            #  if prefixsum is 0, i.e sum(0->i)=0 i.e equal nos of 0's and 1's
            # also if we have seen the prefixsum before then that subarray also has 1's==0's
            if prefixsum==0:
                longest=max(longest,i+1)
            elif prefixsum in seen:
                # j would be seen[prefixsum]
                longest=max(longest, i-seen[prefixsum])
            else:
                seen[prefixsum]=i
        return longest

```

### Subarray product less than k
Given an array of integers nums and an integer k, return the number of contiguous subarrays where the product of all the elements in the subarray is strictly less than k.
Brute Force
```python
class Solution:
    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        # # the idea is same as numSubarraySumK, O(n^2)
        # # Try brute force, i.e all subarrays, inc cnt if product < k
        # cnt=0
        # n=len(nums)
        # for i in range(n):
        #     product=1
        #     for j in range(i,n):
        #         product*=nums[j]
        #         if product<k:
        #             cnt+=1
        # return cnt

```
Optimised
- prefixsum idea wouldn't work here
-think about sliding window

```python
        # Optimised
        # prefixsum idea wouldn't work here
        # think about sliding window
        cnt=0
        l=0
        n=len(nums)
        product=1
        for r in range(n):
            product*=nums[r]
            # if invalid first fix it
            while l<r and product>=k:
                product=product/nums[l]
                l+=1
            # if valid window add all possible subarrays ending at index r, and between l and r, i.e r-l+1
            if product<k:
                cnt+=r-l+1
        return cnt
            
```

### Maximum Product Subarray

Brute Force similar to all prev problems (O(n^2))

Optimised: O(n)


For this question, first think if all numbers are positive, then the maxproduct is product of all elements, if negative numbers then? if even nos of negatives then also product of all numbers. if there are odd nos of negative numbers, we need to exclude one neagtive number, and take the product on either left side or right side of that element.
So we need to calculate prefix and suffix product. Now at each index we find the max of left(prefix) and right(suffix), and then compare this with the global max. this helps us in finding maxProduct

Now if we have zeroes then, what to do? does it change anything ? yes it does. it divides the array into multiple components [-4,2,3,5,0,9,-4,2,-3,0,1,5,2]. here 0 divides the array into 3 components, the maxProduct subarray would be on the left or right og the zeores, so when we calculate the prefix and suffix, we reset the products to 1 when we encounter a 0, so that we start fresh and now see if the new subarray has maxProduct. 




Think intuitively at each index what will be the maxP, consider all cases, negatives (odd and even nos), zeroes. At the end it will be max of either suffix or prefix. In case of zero it makes sense to reset the values back to 1.

```python
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        n=len(nums)
        # the idea here is:
        # if an array has all positives, then the answer would be product of all elements,
        # if has positives and even nos of negatives then also same,
        # if odd nos of negatives then,
        # the answer is either on left side or on the right side right?  [5,-1,8], here answer is either 5 or 8, for [4,-2,-1,-3,4]

        
        # it will not consider one negative and the answer would be on the left of that isolated negatie or on its right. in case of zeroes we reninit the sum as 1

        # so we need prefix and suffix right
        n=len(nums)
        prefix=1
        suffix=1
        for i in range(n):
            prefix*=nums[i]
            suffix*=nums[n-1-i]
            











3`1289cvb nm,.
        # More concise and one iteration
        prefix=1
        suffix=1
        maxP=float('-inf')
        for i in range(n):
            if prefix==0:
                prefix=1
            if suffix==0:
                suffix=1
            prefix*=nums[i]
            suffix*=nums[n-1-i]
            maxP=max(maxP, max(prefix,suffix))
        return maxP
```


the reason why only prefix product would not work:
Now consider nums = [3, -2, 5].

Prefix Loop:

i = 0: prefix = 3, maxP = 3

i = 1: prefix = -6, maxP = 3

i = 2: prefix = -30, maxP = 3

Result: 3

In that second example ([3, -2, 5]), the prefix loop got "stuck" with a negative product because of that single -2. It couldn't "see" that 5 on its own (or 3 on its own) was better than the whole product.







**There are 3 Important array algorithms**
-**Kadane's Algorithm**

In Brute Force, I try all the possible subarrays, I maintian a running sum for subarray starting from each index(i) and ending at variable indices(j) and fianlly check if the maxSum needs to be updated

For Kadane's algorithm which is the optimal version for this problem, the base idea is relatively simple, You iterate once throughout array and you maintian two variables ,one is the global maxSUm and one is the running sum. At each iteration, you add the curr elt's val to the runningsum , now you update the maxsum if required , finally you update the currsum =0 incase if it goes below zero. We do this because if the currsum goes below zero, it wouldnt' help us in finding maxsum anymore, it will rather diminish our returns in future sum too, so its better to start fresh

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




### Majority Element II
Given an integer array of size n, find all elements that appear more than ⌊ n/3 ⌋ times.

Brute Force

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> List[int]:
        # # Brute Force, iterate over the array, store their frequencies in a hashmap, iterate over the hashmap and return the values whose freq>n/3
        # This is an O(n) time and O(n) space solution.
        hashmap={}
        n=len(nums)
        minn=n//3+1
        if n<=2:
            return list(set(nums))
        res=[]
        for num in nums:
            hashmap[num]=hashmap.get(num,0)+1
            if hashmap[num]==minn:
                res.append(num)
        return res
```
Optimised: Extended version of Moore's Voting Algorithm
```python
        # Optimised version would be just O(n) time and O(1) space
        #  We track two majority element in the array using an extended version of Moore's Voting Algorithm
        cnt1,cnt2=0,0
        elt1,elt2=0,0
        n=len(nums)
        for i in range(n):
            if cnt1==0 and nums[i]!=elt2:
                elt1=nums[i]
                cnt1=1
            elif cnt2==0 and nums[i]!=elt1:
                elt2=nums[i]
                cnt2=1
            elif elt1==nums[i]:
                cnt1+=1
            elif elt2==nums[i]:
                cnt2+=1
            else:
                cnt1-=1
                cnt2-=1
        
        # Now we make sure elt1 and elt2 are really majority elements by iterating over the array
        cnt1,cnt2=0,0
        mini=(n//3)+1
        res=[]
        for i in range(n):
            if nums[i]==elt1:
                cnt1+=1
            elif nums[i]==elt2:
                cnt2+=1
        if cnt1>=mini:
            res.append(elt1)
        if cnt2>=mini:
            res.append(elt2)

        return res
```


### Pascal's Triangle

Mainly 3 questions that could be asked around this pattern

Given row and col, return the value at that index from Pascal's triangle
the solution to this would be using nCr i.e n choose r where n would be row-1 and r would be col-1



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

Stack is LIFO Data Structure.
Last-In-First-Out

Implementing Stack using an Array is fairly simple

```python
class MyStack:
    def __init__(self):
        self.stack=[]
        self.top=-1

    def push(self,x:int)->None:
        self.top+=1
        self.stack.append(x)

    def pop(self)->int | None:
        if self.top==-1:
            return None
        else:
            temp=self.stack[self.top]
            self.top-=1
            return temp
    def empty(self)->bool:
        if self.top==-1:
            return True
        else:
            return False
```







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


```python
class Solution:
    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        # # Extreme Naive or Brute Force way to think about this problem is to look for nge for an elt on its right until end of the list, only if not found then also try looking for it from 0 to this elt. else assign -1.
        # # So for every elt we atmax look n-1 elts before we get its nge, this would result in about O(n^2) time
        n=(len(nums))
        nge=[float('inf')]*n
        for i in range(n):
            for j in range(i+1,n):
                if nums[j]>nums[i]:
                    nge[i]=nums[j]
                    break
            if nge[i]==float('inf'):
                # nge still not found, so look in first half
                for j in range(0,i):
                    if nums[j]>nums[i]:
                        nge[i]=nums[j]
                        break
            if nge[i]==float('inf'):
                nge[i]=-1
        return nge

        # The above approach uses O(n^2) time 

        # Optimised verison using stack and linear time
        stack=[]
      
        temp=nums[:]
        temp.extend(nums)
        n=len(temp)
        # now we add nums two times, this way we have the entire array two times and can easily find nge on both the sides, now we use the same logic the one we used for nge 1
        nge=[float('inf')]*(n)
        for i in range(n-1,-1,-1):
            while stack and stack[-1]<=temp[i]:
                stack.pop()
            if stack and stack[-1]>temp[i]:
                nge[i]=stack[-1]
            else:
                nge[i]=-1
            stack.append(temp[i])
        nge=nge[:len(nums)]
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

### Sum of Subarray Minimums

Given an integer array arr, return the sum of the minimum value in each subarray of arr.
Since the answer may be large, return it modulo 10^9 + 7.


```python
class Solution:
    def sumSubarrayMins(self, arr: List[int]) -> int:
        # # Brute force, O(n^2)
        # # Look for all subarrays, find minimum in that subarray and add it to the minsum
        minsum = 0
        n = len(arr)
        modd = (10**9 + 7)
        for i in range(n):
            mini = float('inf')
            for j in range(i, n):
                mini = min(mini, arr[j])
                minsum += mini
        return minsum % modd

```


Optimised
•We want to see how many times an element acts as the minimum in subarrays.
•For each element, find how far we can extend to the left and to the right before hitting a smaller element.
•Use Previous Smaller Element (PSE) and Next Smaller Element (NSE) concepts using a monotonic stack.
•If there are left elements on the left and right on the right where it remains minimum,
then total subarrays with this element as the minimum = left * right.
•Multiply that count with the element value to get its total contribution to the final sum.

```python

class Solution:
    def sumSubarrayMins(self, arr: List[int]) -> int:
        # Optimised version using monotonic stack
        # Each element contributes arr[i] * left * right to the total, 
        # where left = nos of elements greater on left including itself
        # and right = nos of elements greater or equal on right including itself

        n = len(arr)
        mod = (10**9 + 7)

        # find next smaller element (to the right)
        nse = [n] * n
        stack = []
        for i in range(n - 1, -1, -1):
            while stack and arr[stack[-1]] > arr[i]:
                stack.pop()
            nse[i] = stack[-1] if stack else n
            stack.append(i)

        # find previous smaller element (to the left)
        pse = [-1] * n
        stack = []
        for i in range(n):
            while stack and arr[stack[-1]] >= arr[i]:
                stack.pop()
            pse[i] = stack[-1] if stack else -1
            stack.append(i)

        # calculate each element's contribution
        total = 0
        for i in range(n):
            left = i - pse[i]
            right = nse[i] - i
            total += arr[i] * left * right
        return total % mod
```
### Sum of Subarray Ranges
You are given an integer array nums. The range of a subarray of nums is the difference between the largest and smallest element in the subarray.
Return the sum of all subarray ranges of nums.



```python
class Solution:
    def subArrayRanges(self, nums: List[int]) -> int:
        # # brute force, try out all possible subarrays, record max and min and find range and add to the total, takes O(n^2)time
        # total=0
        # n=len(nums)
        # for i in range(n):
        #     maxi,mini=float('-inf'),float('inf')
        #     for j in range(i,n):
        #         # update maxi and mini for this subarray and add to the total the rnage 
        #         maxi=max(maxi,nums[j])
        #         mini=min(mini,nums[j])
        #         total+=maxi-mini
        # return total

        # here the logic is to find maximum and minimum using the logic of sum of subarray minimum and sum of subarray maximum and do max-min and it will give us the answer in linear time O(10N)~= O(N)


        def sumOfSubArrayMaximum(arr:List[int])->int:
            n=len(nums)
            # nge
            stack=[]
            nge=[n]*n
            for i in range(n-1,-1,-1):
                while stack and arr[stack[-1]]<=arr[i]:
                    stack.pop()
                nge[i]=stack[-1] if stack else n
                stack.append(i)

            # pge
            stack=[]
            pge=[-1]*n
            for i in range(n):
                while stack and arr[stack[-1]]<arr[i]:
                    stack.pop()
                pge[i]=stack[-1] if stack else -1
                stack.append(i)

            total=0
            for i in range(n):
                left=i-pge[i]
                right=nge[i]-i
                total+=left*right*arr[i]
            return total
        
        def sumOfSubArrayMinimum(arr:List[int])->int:
            n=len(arr)
            nse=[n]*n
            stack=[]
            for i in range(n-1,-1,-1):
                while stack and arr[stack[-1]]>arr[i]:
                    stack.pop()
                nse[i] = stack[-1] if stack else n
                stack.append(i)
            # similarly pse, start from front
            stack=[]
            pse=[-1]*n
            for i in range(n):
                while stack and arr[stack[-1]]>=arr[i]:
                    stack.pop()
                pse[i] = stack[-1] if stack else -1
                stack.append(i)
            # Now that i have nse and pse time to calculate each element's contribution to the total
            total=0
            for i in range(n):
                left=i-pse[i]
                right=nse[i]-i
                total+=left*right*arr[i]
            return total 
        
        # Now that I have maximums and minimum's contirbution I just subtract mins from max return ans
        return sumOfSubArrayMaximum(nums)-sumOfSubArrayMinimum(nums)

```

### Asteroid Collision




### Remove K digits

Given string num representing a non-negative integer num, and an integer k, return the smallest possible integer after removing k digits from num.

Intuition: Brute force would be to try to find all subsequences if length len(num)-k and see which number is the smallest. This would take exponential time

Now if we think about a number, what makes it bigger or larger, obviously the initial digits, the bigger the initial digit, the bigger the value, so if I want to remove any digits I'd prefer to do it from the front.
I'd use a stack and whenever I'm adding an element, I'd check if it's possible to remove previosue elts that are bigger than this curr elt and k>0 . finally I'd add the number to the stack. Once I'm done iterating over the num, I'd see if my stack is empty the smallest number would be a zero, else I'd aslo chekc if k>0 if this is the case, I didn't find any number greater with which I could replce. so it s better to remove the last digits because they themselves are bigger. I would again check if stack is empty . Now I also need to take care of leading zeores and finally return the ans

```python
class Solution:
    def removeKdigits(self, num: str, k: int) -> str:
        stack=[]
        n=len(num)
        for i in range(n):
            while stack and k>0 and ord(stack[-1])-ord('a')>ord(num[i])-ord('a'):
                stack.pop()
                k-=1
            stack.append(num[i])
        if not stack:
            return '0'
        while k>0:
            stack.pop()
            k-=1
        if not stack:
            return '0'
        # now for leading zeroes
        res=''.join(stack)
        i=0
        while i<len(res) and res[i]=='0':
            i+=1
        res=res[i:]
        if res=="":
            return '0'
        return res
```


## Queue

Queue is a FIFO data structure
First-In-First-Out

Implementing Queue using Array
```python

class MyQueue:
    def __init__(self):
        self.q = []
        self.start = 0
        self.currsize = 0

    def push(self, x: int) -> None:
        self.q.append(x)
        self.currsize += 1
    
    def pop(self) -> int | None:
        if self.currsize == 0:
            return None
        temp = self.q[self.start]
        self.start += 1
        self.currsize -= 1
        return temp

    def top(self) -> int | None:
        if self.currsize == 0:
            return None
        return self.q[self.start]

    def empty(self) -> bool:
        return self.currsize == 0

    def size(self) -> int:
        return self.currsize


```

## Linked List

### Structure
- Linear data structure (chain of nodes)
- Each node contains: **data** and **next** pointer
- Elements not stored contiguously (unlike arrays)

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
If we observe carefully, we can clearly see that either left side or right side is sorted. Why? (because we are given a sorted array, it might be rotated, so there would be a pivot point in one of the halfs, the rest part would definitely be sorted)
Now binary search only works on a sorted array, so we check which portion is sorted, after that we check if target lies within the left side or not , if yes we do binary search on left side, else we move to the right side


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
                    # now check if target lies on left side, then prune the search space by eliminating right half since target lies on left
                    if nums[l]<=target<=nums[mid]:
                        r=mid-1
                    # else it lies on right side
                    else:
                        l=mid+1
                else:
                    # right is sorted
                    # now similarly check for right half sorted case. check if target exists on this sorted right part else got to left
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

### Minimize Max Distance to Gas Station

Given a sorted array arr of size n, containing integer positions of n gas stations on the X-axis, and an integer k, place k new gas stations on the X-axis.

The new gas stations can be placed anywhere on the non-negative side of the X-axis, including non-integer positions.

Let dist be the maximum distance between adjacent gas stations after adding the k new gas stations.

Find the minimum value of dist.

Your answer will be accepted if it is within 1e-6 of the true value.


Example 1

Input: n = 10, arr = [1, 2, 3, 4, 5, 6 ,7, 8, 9, 10], k = 10

Output: 0.50000

Explanation:

One of the possible ways to place 10 gas stations is [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10].

Thus the maximum difference between adjacent gas stations is 0.5.

Hence, the value of dist is 0.5.

It can be shown that there is no possible way to add 10 gas stations in such a way that the value of dist is lower than this.


```python
class Solution:
    def minimiseMaxDistance(self, arr, k):
        n = len(arr)
        # Store: (-max_distance_in_segment, segment_length, num_stations_placed)
        heap = []
        
        for i in range(1, n):
            segment_length = arr[i] - arr[i-1]
            # Initially 0 stations in each segment
            # Max distance = segment_length / (0 + 1) = segment_length
            heapq.heappush(heap, (-segment_length, segment_length, 0))
        
        # Place k gas stations
        for _ in range(k):
            # Get segment with maximum distance
            max_dist, original_length, stations = heapq.heappop(heap)
            
            # Add one more station to this segment
            stations += 1
            
            # New max distance in this segment after adding one station
            new_max_dist = original_length / (stations + 1)
            
            # Push back with updated count
            heapq.heappush(heap, (-new_max_dist, original_length, stations))
        
        # The answer is the maximum distance among all segments
        return -heap[0][0]

```
Above solution takes O(nlogn +klogn)







## Sliding Window

### 424. Longest Repeating Character Replacement

You are given a string s and an integer k. You can choose any character of the string and change it to any other uppercase English character. You can perform this operation at most k times.

Return the length of the longest substring containing the same letter you can get after performing the above operations.

Solution:
We are trying to achieve the longest substring with only repeating chars. You can perform atmost k operations to replace other character in the string.
If we take a step back, what are we trying a substring with a char who's most frequent and the remaining chars should not exceed k.
So instead of worrying about replacing, we just maintain the window in this manner. We start from front and add each elt to the window and maintain a freq map. now if the length of window - most freq elt seen so far's freq >k then this is surely an invalid winodw. so we move l and update freq until we have a valid window. Now we calculate maximum and update when required

```python
class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        hashmap=defaultdict(int)
        n=len(s)
        l=0
        r=0
        maxi=0
        while r<n:
            hashmap[s[r]]+=1
            # now if my window l:r has an elt with max freq 
            mostfreq=max(hashmap, key=hashmap.get)
            # windowLength=r-l+1
            while r-l+1-hashmap[mostfreq]>k:
                hashmap[s[l]]-=1
                l+=1
                mostfreq=max(hashmap, key=hashmap.get)
                # invalid window, fix it
            maxi=max(maxi,r-l+1)
            r+=1
        return maxi

```

### 1248. Count Number of Nice Subarrays

Given an array of integers nums and an integer k. A continuous subarray is called nice if there are k odd numbers on it.

Return the number of nice sub-arrays.

Brute: Try all subarrays 

Optimised: Use hashmap to store how many times particular odd numbers encountered (3odd nos appeared 2 times, 1 odd number appeared 4 times)
```python
class Solution:
    def numberOfSubarrays(self, nums: List[int], k: int) -> int:
        # # brute force
        # # try all subarrays
        # cnt=0
        # n=len(nums)
        # for i in range(n):
        #     oddCnt=0
        #     for j in range(i,n):
        #         if nums[j]%2!=0:
        #             oddCnt+=1
        #         if oddCnt==k:
        #             cnt+=1
        # return cnt

        # similar to runningsum, what if I maintain cnt of odd nos
        n=len(nums)
        cnt=0
        currOddCnt=0
        hashmap=defaultdict(int)
        hashmap[0]=1
        for i in range(n):
            if nums[i]%2!=0:
                currOddCnt+=1
            if currOddCnt-k in hashmap:
                cnt+=hashmap[currOddCnt-k]
            hashmap[currOddCnt]+=1
        return cnt

```


### 1358. Number of Substrings Containing All Three Characters

Given a string s consisting only of characters a, b and c.
Return the number of substrings containing at least one occurrence of all these characters a, b and c.

```python

class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        # # brute force, try all substrings and cnt++ if valid
        # cnt=0
        # n=len(s)
        # for i in range(n):
        #     a,b,c=0,0,0
        #     for j in range(i,n):
        #         if s[j]=='a':
        #             a+=1
        #         elif s[j]=='b':
        #             b+=1
        #         else:
        #             c+=1
        #         if a>0 and b>0 and c>0:
        #             cnt+=1
        # return cnt

        # optimal solution, we use hashing to store indexes, idea is at each index, how much left we can go so that the substring becomes valid, now all the substrings that can be formed with that as the ending aprt will be added to the count
        lastseen=[-1,-1,-1]
        n=len(s)
        cnt=0
        for i in range(n):
            lastseen[ord(s[i])-ord('a')]=i
            if min(lastseen)!=-1:
                cnt+=min(lastseen)+1
        return cnt


```












## Trees

### Traversals

There are 2 main traversal types for a Tree:
1) DFS (Depth First Search):


Intuition for these traversals is fairly simple, just do what we want on a node and its parent in a recurring fashion. so here we want root's value then left child's value and finally the right child's value. So whenever I have a node(root node), I add to my result, now I need my left child's value, but wait that itself is a tree, so I call recursive function to do the same thing (so now I add left subtree's root value to the result, then add left subtree's left child and then right child to this result and return this to the root of the tree). Well now I have my root's value, left child(left subtree and its value) now I do the same for right subtree and then finally return the res.

- Pre-Order Root, Left, Right
Time Complexity is O(N)
Space Complexity is O(height of tree/logn) in worst case O(N)
Here the space used is auxiliary space that is used by recursion stack to keep function call which is mostly logn/ height of tree. In rare cases, when the tree is skewed( linear like), in those cases height of the tree turns out to be n and so does the space O(N).

```python

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        # Recursion based
        res=[]
        if not root:
            return []
        res.append(root.val)
        res+=self.preorderTraversal(root.left)
        res+=self.preorderTraversal(root.right)
        return res


# Iterative, since dfs algorithm uses a stack data structure
        stack=[]
        res=[]
        if not root:
            return res
        stack.append((root))
        # Here we just invert the logic, preorder means root, left, right, here we do root, then we check first for right and then left, because stack is a LIFO ds, so adding right and then ledt to the top makes sense, so when we pop the stack we first get left and then right. so it follows the order Root, left, right.
        while stack:
            node=stack.pop()
            res.append(node.val)
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
        return res

```

- Post-Order Left, Right, Root  

Time and Space Complexity same as other traversals O(N) and O(logn) space in best case and O(N) worst case.


```python

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        res=[]
        if not root:
            return res
        res=self.postorderTraversal(root.left)
        res+=self.postorderTraversal(root.right)
        res.append(root.val)
        return res
```



- In-Order Left, Root, Right

```python

class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        # left,root,right
        res=[]
        if not root:
            return []
        res=self.inorderTraversal(root.left)
        res.append(root.val)
        res+=self.inorderTraversal(root.right)
        return res


```



2) BFS (Breadth First Search):
Level Order Traversal is simple, below is a modified version where we just reverse alternate levels to attain zig-zag level order traversal

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        res=[]
        if not root:
            return res
        q=deque()
        q.append(root)
        boolean=True
        while q:
            level=[]
            for i in range(len(q)):
                node=q.popleft()
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
                level.append(node.val)
            if not boolean:
                level.reverse()
            res.append(level)
            boolean=not boolean 
        return res
```


### Boundary Traversal of a Binary Tree (Anticlockwise)

Boundary Traversal means we include all the nodes that on the exterior of B.T, we start initially with root, go down to the leftmost node possible, we do root.left, root=root.left until we have a left node. if not then we do root.right and again continue until we reach leaf node. Now we stop. We perform an inorder traversal to get all the leaf nodes, following that we start again from root.right and go to right most node until we reach leaf. we store this right exterior nodes in another data structure and then reverese them and add to the res

```python
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val=val
#         self.left=left
#         self.right=right

class Solution:
    def isLeaf(self, root:TreeNode):
        return not root.left and not root.right
            
    def addLeftBoundary(self, root,res):
        curr=root.left
        while curr:
            if not self.isLeaf(curr):
                res.append(curr.val)
            if curr.left:
                curr=curr.left
            else:
                curr=curr.right
    

    def addLeavesInorder(self, root,res):
        curr=root
        if self.isLeaf(curr):
            res.append(curr.val)
            return
        self.addLeavesInorder(curr.left)
        self.addLeavesInorder(curr.right)
        

    
    def addRightBoundary(self, root,res):
        curr=root.right
        temp=[]
        while curr:
            if not self.isLeaf(curr):
                temp.append(curr.val)
            if curr.right:
                curr=curr.right
            else:
                curr=curr.left
        res.extend(temp[::-1])

    def boundaryTraversal(self,root):
        res=[]
        if not root:
            return res
        if not self.isLeaf(root):
            res.append(root.val)
        
        self.addLeftBoundary(root,res)
        self.addLeavesInorder(root,res)
        self.addRightBoundary(root,res)
        return res
```

### Vertical Order Traversal

```python

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def verticalTraversal(self, root: Optional[TreeNode]) -> List[List[int]]:
        # I use nested dictionary to store x:y:nodes,
        hashmap=defaultdict(lambda: defaultdict(list))
        res=[]
        if not root:
            return res
        # q(node,(x,y))

        q=deque()
        q.append((root,(0,0)))
        while q:
            for i in range(len(q)):
                node,(x,y)=q.popleft()
                if node.left:
                    q.append((node.left,(x-1,y+1)))
                if node.right:
                    q.append((node.right,(x+1,y+1)))
                hashmap[x][y].append(node.val)
                # hashmap store for each x diff y
        # now from this hashmap, I oragnize it in vertical lines
        for x,value in sorted(hashmap.items()):
            temp=[]
            for y,node_val in value.items():
                temp.extend(sorted(node_val))
            res.append(temp)
        return res


        
```


### Top View of a Binary Tree

This can be boiled down if we use just the idea of Vertical order traversal of lines and levels. It is easier to use level order traversal than the recursive approach.

I initially start with the root node at line 0 in the queue, now I iterate over the queue and store the line-> node in hashmap if the key(line) doesn't exist in the hashmap. Now for this node, I check the left and right child and add them to the queue. For left child I add (node.left, node's line - 1) and for the right (node.right, node's line + 1)

```python   
class Solution:
    def TopView(root)->List[int]:
        topview=[]
        if not root:
            return topview
        q=deque()
        q.append((root,0))
        hmap={}
        while q:
            node,line=q.popleft()
            if line not in hmap:
                hmap[line]=node.val
            if node.left:
                q.append(node.left, line-1)
            if node.right:
                q.append(node.right, line+1)
        
        # Now the hashmap stores line->node
        for line,node in sorted(hmap.items()):
            topview.append(node)
        return topview
```





### Bottom View of Binary Tree
This is same as top view, only difference here is we update the map every time, so we get the node's value for a line at the last level

```python   
class Solution:
    def BottomView(root)->List[int]:
        bottomview=[]
        if not root:
            return bottomview
        q=deque()
        q.append((root,0))
        hmap={}
        while q:
            node,line=q.popleft()
            # here we update the map every time, so we get the node's value for a line at the last level
            hmap[line]=node.val
            if node.left:
                q.append(node.left, line-1)
            if node.right:
                q.append(node.right, line+1)
        
        # Now the hashmap stores line->node
        for line,node in sorted(hmap.items()):
            bottomview.append(node)
        return bottomview
```

### All paths from Root to Leaf

```python
class Solution:
    def rootToNodePaths(self, root:TreeNode)->List[List[int]]:
        # stores all the paths
        paths=[]
        # store one path at a time
        path=[]

        def findPaths(root):
            # base case
            # When do i stop, once I reach a null node
            if not root:
                # stop cant go further
                return
            # what can i do from a node, i add it to my path
            path.append(root.val)
            # now I had a valid node so ia dded to my path if no further nodes, then this is a leaf node, i add this to my paths
            if not root.left and not root.right:
                paths.append(path[:])

            # now I can explore my left path or right path
            if root.left:
                findPaths(root.left)
            if root.right:
                findPaths(root.right)
            # Now I used root and all its paths i.e on left and on right, now i remove it
            path.pop()

```

### Find Root to Node Path

Given root of a Binary Tree and a node(val) find root to node path

```python
class Solution:
    def rootToNodePaths(self, root:TreeNode, node:int)->List[int]:
        if not root:
            return []
        path=[]
        def searchNode(root,node):
            # base case
            if not root:
                return False
            # i add the curr root to the path
            path.append(root.val)
            # if I reach the target node, i return 
            if root.val==node:
                return True
            # Else I explore left and right path starting fromthis node
            if searchNode(root.left,node)==True:
                return True
            if searchNode(root.right,node)==True:
                return True
            path.pop()
            return False
            
        if searchNode(root,node)==True:
            return path

```




### Maximum Path Sum

A path in a binary tree is a sequence of nodes where each pair of adjacent nodes in the sequence has an edge connecting them. A node can only appear in the sequence at most once. Note that the path does not need to pass through the root.
The path sum of a path is the sum of the node's values in the path.
Given the root of a binary tree, return the maximum path sum of any non-empty path.


```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        maxi=float('-inf')
        def pathsum(root):
            nonlocal maxi
            if not root:
                return 0
            leftsum=max(0,pathsum(root.left))
            rightsum=max(0,pathsum(root.right))
            maxi=max(maxi,root.val+leftsum+rightsum)
            return root.val+ max(leftsum,rightsum)
        
        pathsum(root)
        return maxi

```


### Maximum Width Of Binary Tree

We use indexing logic from segment trees concept i.e Given a node at position i, the left node will be 2*i+1 and the right node will be 2*n+2. This works when we start from 0 based indexing, for one based indexing we do 2*i and 2*i+1. There's an edge case where this fails, for skewed trees, so for that at eachlevel we subtract minimum index of that level from the node's position. This way all the nodes at each level always starts from 1.


```python

class Solution:
    def widthOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        q=deque()
        q.append((root,0))
        maxi=0
        while q:
            level=[]
            min_idx=q[0][1]
            for _ in range(len(q)):
                node,pos=q.popleft()
                pos-=min_idx
                if node.left:
                    q.append((node.left,2*pos+1))
                if node.right:
                    q.append((node.right,2*pos+2))
                level.append(pos)
            print(level)
            # max(pos)-min(pos)
            maxi=max(maxi,((max(level)-min(level)+1)))
        return maxi
```

### Construct Binary Tree from Preorder and Inorder

```python
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:

        # lets build a hashmap for inorder elements with their indices for fast lookups
        hmap={}
        for idx,val in enumerate(inorder):
            hmap[val]=idx


        def f(preS, preE, inS, inE):
            if preS>preE or inS>inE:
                return None
            # create root for this step
            root=TreeNode(preorder[preS])
            idx=hmap[root.val]
            numsleft=idx-inS
            # all the elts or indices to the left of this root index would be aprt fo leftsubtree 
            left=f(preS+1,preE+numsleft,inS,idx-1)
            right=f(preS+numsleft+1 ,preE,idx+1,inE)
            root.left=left
            root.right=right
            return root
        
        root=f(0,len(preorder)-1,0,len(inorder)-1)
        return root
```




### Morris Inorder Traversal (Left Root Right)

```python

class Solution:
    def MorisInorderTraversal(root:Optional[TreeNode])->List[int]:
        # the idea here is to connect rightmost node of every left subtree to the root for traversing back
        # we start initially with root, now there are 2 possibilities, there can be a node.left or not,
        # if not node.left and then according to inorder(left,root,right) we can simply add the root value to our traversal list and move onto root.right.
        # But what if there is a node.left, then we have to explore that subtree first, but how will we return to the root and then explore right subtree once we finished exploring the left? so we use a little trick, whenever a left subtree/node exists for a node, we use a prev pointer and point to this left subtree, now we keep moving this pointer to the rightmost node of this left subtree and then attach that to our root/current node, so once done exploring left we can easily move to the root and then explore right.
        # One edge case here is we need to check if the rightmost node of the left subtree already points to our root/curr node it implies that we have already visited the left subtree and now we remove this connection, add the root's val to our traversal and move onto explore the right
        inorder=[]
        if not root:
            return inorder
        curr=root 
        while curr:
            # 1st case no left
            if not curr.left:
                inorder.append(curr.val)
                curr=curr.right
            else:
                # left node/subtree exists
                # so get to the righmost node and connect it to root and then change curr to curr.left, so we can come back to it once done with the left
                prev=curr.left
                while(prev.right and prev.right!=curr):
                    prev=prev.right
                # now the prev.right can either point to a null value or to the curr node itself 
                if prev.right==None:
                    # attach it to curr
                    prev.right=curr
                    # now explore the left subtree
                    curr=curr.left
                else:
                    # prev.right points to the curr node, i.e the left is already explored
                    # break the connection
                    prev.right=None
                    inorder.append(curr.val)
                    # go to right
                    curr=curr.right
        return inorder
```


### Morris Preorder Traversal (Root Left Right)

This is similar to the previous with just a little change in when we add the curr's val to the traversal
Here we add the curr node's value when we attach the rightmost node to the curr and hten explore the left subtree( root left right)


```python

class Solution:
    def MorisPreorderTraversal(root:Optional[TreeNode])->List[int]:
        preorder=[]
        if not root:
            return preorder
        curr=root 
        while curr:
            # 1st case no left
            if not curr.left:
                preorder.append(curr.val)
                curr=curr.right
            else:
                # left node/subtree exists
                # so get to the righmost node and connect it to root and then change curr to curr.left, so we can come back to it once done with the left
                prev=curr.left
                while(prev.right and prev.right!=curr):
                    prev=prev.right
                # now the prev.right can either point to a null value or to the curr node itself 
                if prev.right==None:
                    # attach it to curr
                    prev.right=curr
                    preorder.append(curr.val)
                    # now explore the left subtree
                    curr=curr.left
                else:
                    # prev.right points to the curr node, i.e the left is already explored
                    # break the connection
                    prev.right=None
                    # go to right
                    curr=curr.right
        return preorder
```

### Flatten Binary Tree to Linked List

```python
class Solution:
    def flatten(self, root:TreeNode)-> None:
        # modify in place the binary tree into a linked list like structure, if we carefully observe the linkedlist follows the preorder traversal of a binary tree. So we have a root and it has left and right subtrees, now if we want the preorder (root,left,right), we dont do anything to root, we simply go to left but we will also need to come back to right node, so we go to the rightmost node fo the left subtree and connect it to the right subtree's start node, and then we assign the root's right node to root's left node, and make left node null. so now we have a root with left node = null and right node pointing to the left usbtree, the left subtree 's last node points to th right subtree's first node, now we move the curr pointer to curr.right and perform the same step, until we reach the last node/null

        # This is the most optimal solution using O(N) time and O(1) space
        curr=root
        while curr:
            if curr.left:
                # i need to find rightmost node and connect it to the first node of right subtree/curr's right 
                prev=curr.left
                while prev.right:
                    prev=prev.right
                prev.right=curr.right
                curr.right=curr.left
                curr.left=None
            curr=curr.right


        # Another solution using a stack O(N) time and O(N) space
        if not root:
            return None
        stack=[root]
        while stack:
            curr=stack.pop()
            if curr.right:
                stack.append(curr.right)
            if curr.left:
                stack.append(curr.left)
            curr.left=None
            if stack:
                curr.right=stack[-1]    



```


### Binary Search Tree

This type of tree holds the property where all the elements on the left subtree < Root < All elements on the right subtree 
also this property holds for all the subtree as well


### Search in a BST
You just recursively check for each node, if it equals to the curr node, return it, else move left or right based on the key you're searching for. This is mostly an O(logn) or O(height) solution unless a skewed tree

```python
class Solution:
    def search(root:TreeNode, key:int)->Optional[TreeNode]:
        if not root:
            return None
        temp=root
        while temp:
            if temp.val==key:
                return temp
            elif temp.val>key:
                # go left
                temp=temp.left
            else:
                # go right
                temp=temp.right
        # if here that means we reached a null node and the node doesn't exist return None
        return None
```

### Ceil in a BST
Given a Root of a BST and a key, return the ceil for the node
Ceil should be the smallest key/node greater than the key provided, if it exists, else `None`
```python
class Solution:
    def Ceil(self, root:Optional[TreeNode], key:int) -> Optional[TreeNode]:
        if not root:
            return None
        temp=root
        ceil=None
        while temp:
            if key==temp.val:
                return temp
            elif key<temp.val:
                ceil=temp.val
                temp=temp.left
            else:
                temp=temp.right
        return ceil

```

### Floor in a BST

Given a Root of a BST and a key, return the floor for the node
Floor should be the greatest key/node smaller than the key provided, if it exists, else `None`
```python
class Solution:
    def Floor(self, root:Optional[TreeNode], key:int) -> Optional[TreeNode]:
        if not root:
            return None
        temp=root
        floor=None
        while temp:
            if key==temp.val:
                return temp
            elif key>temp.val:
                floor=temp.val
                temp=temp.right
            else:
                temp=temp.left
        return floor

```

### Insertion in BST

```python
class Solution:
    def insertionBST(self, root:Optional[TreeNode], key:int) -> TreeNode:
        if not root:
            return TreeNode(key)
        temp=root
        prev=None
        while temp:
            prev=temp
            if key<temp.val:
                temp=temp.left
            else:
                temp=temp.right
        # Now prev is either smaller or greater than the key
        if prev.val<key:
            prev.right=TreeNode(key)
        else:
            prev.left=TreeNode(key)
        return root

```




### Deletion in BST

```python
class Solution:
    def deleteNode(self, root: Optional[TreeNode], key: int) -> Optional[TreeNode]:

        def helper(node):
            # when we get a node here that node is to be deleted, there are three cases: no right child, no left child, or both present
            if not node.left:
                return node.right
            elif not node.right:
                return node.left
            # now here it means both child present, so we smartly will return the left subtree and attach the right subtree to the rightmost node on the left subtree
            
            rightchild=node.right
            lastright=node.left
            while lastright.right:
                lastright=lastright.right
            # now attach this to rightchild
            lastright.right=rightchild
            return node.left

        


        if not root:
            return None
        if root.val==key:
            return helper(root)
        dummy=root
        while dummy:
            # check if key is on left or right recursively 
            if dummy.val>key:
                # key is on left part
                # check if it is on dummy.left, else move to dummy.left 
                if dummy.left and dummy.left.val==key:
                    dummy.left=helper(dummy.left)
                    break
                else:
                    dummy=dummy.left
            else:
                # key is on right part
                if dummy.right and dummy.right.val==key:
                    dummy.right=helper(dummy.right)
                    break
                else:
                    dummy=dummy.right
        return root
        

```


## Graphs


Directed and Undirected Graphs
Connected Components
n nodes and m edges

Now given n nodes and m edges, we can represent it in adjacency matrix or adjacency list

For shortest path or dijsktra's algorithm if the edge weights are equal (unit weights), we can omit using a PQ, and simple use a regular queue. Because it will be sorted in default fashion because at each level, we unifromly increase by 1, there are not random edge weights that increase the distacne. If that was the case then we might have used PQ. That doesn't implies Q wouldn't work in that case, i twould work but a bit extra redundant calculations. 


### BFS

Breadth First Search
Level Traversal

We use a queue data structure and a visited array of size equal to nos of nodes.
We add the starting node to the queue and mark it as visited. now while queue is not empty, we pop out a node and explore all the nieghbors if they are not visited or explored

Storing graph can be done in either adjacency matrix (O(n^2)) or adj list O(2E) 2*edges for undirected. In directed graph space will be O(E)

The intuition for BFS is same as the BFS for trees, but here the catch is there is no left and right child, here there are neighbors, so we iterate over all the neighbors and add them to the queue

```python

def bfs(int V, adj:List ):
  
    q=deque()
    q.append(0)
    visited=[0]*(n+1)
    bfs=[]
    while q:
        node=q.popleft()
        visited[node]=1
        for nei in adj[node]:
            if not visited(nei):
                q.append(nei)
        bfs.append(node)
    return bfs
```



### DFS
Recursive way

```python

def dfsTraversal(int V, adj:List):
    traversal=[]
    visited=[0]*V+1
    visited[0]=1
    def dfs(node):
        visited[node]=1
        traversal.append(node)
        for nei in adj[node]:
            if not visited(nei):
                dfs(nei)
    
    dfs(0)
    return traversal

```

Time complexity rememeber:
degree of node is nos of edges connected to a node
sum of all degrees of all the nodes is 2E


### Bipartite Graph

Imp note:
Linear graphs are always bipartite.
If a cycle exists, then if length of cycle is even then bipartite, if odd lenght cycle then cannot be biparite

Given a graph, if you can color the graph using 2 colors such that no two adjacent nodes have same color, then it is a bipartite graph.
We use color array equal to nos of nodes as -1 init and use 0 and 1 as the colors.
Now we start i from 0 to nos of nodes, if color==-1 and then I do dfs traversal on that node. Now assign that node a color and then explore it neighbor and assign the opposite color. Now if this constraint doesn't work then return False else fianlly return True



```python

class Solution:
    def biparite(V:int, adj:List[List[int]])->bool:
        
        color=[-1]*V
        def dfs(node,col):
            color[node]=col
            for nei in adj[node]:
                if not color[nei]:
                    if dfs(nei,not col)==False:
                        return False
                # elif color exists
                elif color[nei]==col:
                    return False
                    # cannot have same color
            return True

        

        for i in range(V):
            if color[i]==-1:
                if dfs(i,0)==False:
                    return False
        return True



```

### Topo Sort

Topo Sort Only works on DAG (Directed Acyclic Graph):
Linear Ordering of vertices such that is there is an edge between u and v, u appears before v in the ordering
Topo Sort does not work on undirected and also cyclic graph, because it is not possible to have such ordering 1-2 here 1->2 and 2->1 , both are not possible, so only works for directed. Also if a cycle exists then 1->2->3->1 then no possible linear ordering.


**DFS Style**

```python
class Solution:
    def TopoSort(self, V:int, adj:List[List[int]])->bool:
        
        def dfs(node):
            visited[node]=1
            path_visited[node]=1
            for nei in adj[node]:
                if not visited[nei]:
                    if dfs(nei)==True:
                        return True
                elif path_visted[nei]:
                    return True
            path_visited[node]=0
            toposort.append(node)
            return False
                

        toposort=[]
        # connected components logic
        visited=[0]*V
        path_visited=[0]*V
        for i in range(V):
            if not visited[i]:
                # dfs return True if a cycle exists
                if dfs(i)==True:
                    return False
        return True
```




### MST

You are given an undirected weighted graph, build a mst or return mst weight

A spanning tree is a tree/graph that has N nodes and N-1 edges, and it is such that all the nodes are connected and reachable from each other. 
Multiple spanning tree possible.  spanning tree is just a tree that connects all the nodes with n-1 edges

A minimum spanning tree is the spanning tree of the graph with minimum total edge weight

To ways to find MST, prim's algorithm and kruskel's algorithm

### Prim's Algorithm

It uses a Priority Queue(heap)(weight,node,parent) and visited array
Uses f=greedy algorithm
Starts with any node (0,0,-1)
Now it checks all its neighbirs, if not visited adds it to the queue.
When a pair is popped from pq, it is checked if visited or not, if not then first visit it, add its weight to the mst weight and explore neighbors. if already visited continue. Continue this until PQ empty  


```python

class Solution:
    def spanningTree(self, V, adj):
        # prims algorithm uses a priority queue and a visited array
        # init all the vertices are not visited and the queue starts off with
        # first vertice, lets say 0 for our case
        heap=[]
        visited=[0]*V
        # (wt,node)
        heapq.heappush(heap,(0,0,-1))
        mswt=0
        mst=[]
        while heap:
            wt,node,par=heapq.heappop(heap)
            if visited[node]:
                continue
            visited[node]=1
            mswt+=wt
            if parent != -1:
                mst.append([node,par])
            # explore the neighbors
            for (nei,w) in adj[node]:
                if not visited[nei]:     
                    heapq.heappush(heap,(w,nei,node))
        return mswt

```





### Disjoint Set

To check if two nodes are connected together or are in smae component.

Brute force would be to traverse the component and check if it exists. this would be linear or O(V+E).
Here comes the concept of Disjoint Set, they answer in O(1).  
two important functions findParent() and union(by rank or by size)

**Union by Rank**

Initialise rank array with 0's and parent array with nodes itself. 1 is parent of 1, 7 is parent of 7, etc
Tracks the rank (approximate height/depth) of each tree
Rank only increases when both ranks are equal


Union(u,v):
1) find ultimate parent of u and v i.e pu and pv
2) if pu == pv return (because already together)
3) find rank of pu and pv
4) connect smaller rank to larger rank always  
5) if equal rank then increase rank of the tree on which you attach the other tree

**Union by Size**

1) find ultimate parent of u and v i.e pu and pv
2) if pu == pv return (because already together)
3) find size of pu and pv
4) connect smaller size to larger size always  
5) increase size of larger tree by size of smaller tree

Tracks the size (number of nodes) in each tree/set
Always attach the smaller tree to the larger tree
Always update size: size[larger] += size[smaller]
Size gets updated every time a union happens


```python
class DisjointSet:
    def __init__(self,n):
        self.rank=[0]*n
        self.parent=[i for i in range(n)]
        self.size=[1]*n

    def findParent(self,u):
        if self.parent[u]==u:
            return u
        #  using path compression
        self.parent[u]=self.findParent(self.parent[u])
        return self.parent[u]
       
        # parent[u]=findParent(parent[u])
        # return parent[u]
        # this sets the top guy as parent of each and every node in the path, so all these nodes will be directly connected to the root/ultimate parent
    
    def unionbyRank(self,u,v)->None:
        # if both are in same component/tree/grpah no need to do union
        # check by checking ultimate parents of both using findParent method
        pu=self.findParent(u)
        pv=self.findParent(v)
        if pu==pv:
            return
        # else union their ultimate parents based on rank
        if self.rank[pu]>self.rank[pv]:
            self.parent[pv]=pu
        elif self.rank[pu]<self.rank[pv]:
            self.parent[pu]=pv
        else:
        # in equal case any can be attached and rank increases
            self.parent[pu]=pv
            self.rank[pv]+=1
    
    def unionbySize(self,u,v)->None:
        # if both are in same component/tree/grpah no need to do union
        # check by checking ultimate parents of both using findParent method
        pu=self.findParent(u)
        pv=self.findParent(v)
        if pu==pv:
            return
         # else union them based on size
        if self.size[pu]>self.size[pv]:
            self.parent[pv]=pu
            self.size[pu]+=self.size[pv]
        else:
            self.parent[pu]=pv
            self.size[pv]+=self.size[pu]
        # in equal case any can be attached 


```





### Kruskel's Algorithm

Sort all the edges according to the weight
Use Disjoint Set 
Initially all nodes are graph itself. we start iterating over the sorted edges and see if they are connected (check ultimate parents to see if they are in same component), update weight along the way, if not we union them and move forward


```python

class DisjointSet:
    pass
    # ....
    # ....
    # ....

class Solution:
    def spanningTree(self, V, adj):
        # kruskels algorithm
        # given nos of nodes and adj list
        # i needed sorted edges based on weights
        edges=[]
        for u in range(V):
            for (nei,wt) in adj[u]:
                edges.append((wt,u,nei))
        # now i have all edges in one list, let's sort them
        edges.sort()
        # edges store all edges to create mst
        mst_edges=[]
        # mst weight 
        mst_wt=0
        # now i got through edges and check if i already 
        # have the node u and v connected, if yes i dont consider 
        # that edge, else i add the edge to mst edge and mstwt
        # I create a disjoint set instance 
        dsu =DisjointSet(V)
        for ed in edges:
            (wt,u,v)=ed
            pu=dsu.findParent(u)
            pv=dsu.findParent(v)
            if pu!=pv:
                dsu.unionbySize(u,v)
                mst_edges.append((u,v))
                mst_wt+=wt
        return mst_wt

```


### Cycle Detection in a Directed Graph

What does cycle in a graph actually means? A cycle is nothing more than you reach a same node twice on the same path. Remember this same path. If I start from node A and reach back to node A in the same path, then there is a cycle, else no cycle.
How do I find that in a directed graph? Only using a visited array doesn't help here. Because it is possible to reach a node twice usign different paths, so it is important to use another path_visited array 

**DFS Style**

```python
class Solution:
    def hasCycleDirectedGraph(self, V:int , adj:List[List[int]])->bool:
        # visited array helps to avoid visiting same node again
        # path-visited helps to see if a node is visited on same path, only then cycle, when we backtrack we mark the node as unvisited in path_visited, but not in visited
        visited=[0]*V
        path_visited=[0]*V
        
        def dfs(node):
            visited[node]=1
            path_visited[node]=1
            for nei in adj[node]:
                if not visited[nei]:
                    if dfs(nei)==True:
                        return True
                elif path_visited[nei]:
                    return True
            path_visited[node]=0
            return False
        

        # check for all components, if one has cycle return True right away
        for i in range(V):
            if not visited[i]:
                if dfs(i)==True:
                    return True
        return False

```
**BFS Style**

This would be using Kahn's Algorithm, i.e Topo Sort. If a valid topo sort found then no cycle, else a cycle is present.
Why does this work? We know Topo Sort works only for Directed Acyclic Graph. So in our directed graph, a cycle exists then we can't find a valid topo sort ordering

```python
class Solution:
    def hasCycleDirectedGraph(self, V:int, adj:List[List[int]])->bool:
        indegree=[0]*V
        for u in range(len(adj)):
            for nei in adj[u]:
                indegree[nei]+=1
        q=deque
        toposort=[]
        # now add all the nodes with 0 indegree to q
        for i in range(V):
            if indegree[i]==0:
                q.append(i)
        
        while q:
            node=q.popleft()
            toposort.append(node)
            for nei in adj[node]:
                indegree[nei]-=1
                if indegree[nei]==0:
                    q.append(nei)
        # if len(toposort)!= num of nodes, then cycle exists
        return len(toposort)!=V
```




### Cycle Detection in an Undirected Graph

This one's fairly simple. I just do a standard dfs where I keep track of node and its parent. Whenever I visit the neighbors of a node in a dfs and the node is not visited, I call a dfs on that nei. If that dfs calls return True, I return True and break out immediately, because there is a cycle. Else case if the nei is visited, now if the nei is parent of the node, it is bound to be visited, so we don't care, but if it's not the parent and still visited, then there definitely is a cycle. I return True. Finally after the dfs call is complete and we didn't find any cycle, it is safe to assume that there are no cycles in this undirected graph. We also need to check or start this dfs call by using the visited array so that is graph has multiple components, we account for each of them, and if any one has a cycle, we return True immediately.

**DFS Style**

```python
class Solution:
    def hasCycleUndirectedGraph(self, V:int , adj:List[List[int]])->bool:
        visited=[0]*V
        def dfs(node,par):
            visited[node]=1
            for nei in adj[node]:
                if not visited[nei]:
                    if dfs(nei,node)==True:
                        return True
                # visited
                elif nei!=par:
                    return True
            return False

        for i in range(V):
            if not visited[i]:
                if dfs(i,-1):
                    return True
        return False


```

**BFS Style**

```python
class Solution:
    def hasCycleUndirectedGraph(self, V:int , adj:List[List[int]])->bool:
        visited=[0]*V

        def bfs(node,par):
            visited[node]=1
            q=deque()
            q.append((node,-1))
            while q:
                node,par=q.popleft()
                for nei in adj[node]:
                    if not visited[nei]:
                        visited[nei]=1
                        q.append((nei,node))
                    # visited
                    elif nei!=par:
                        return True
            return False
                    
                    

        for i in range(V):
            if not visited[i]:
                if bfs(i,-1):
                    return True
        return False



```

### Shortest Path in DAG (Directed Acyclic Graph)
Given a graph in form of adj list with (node,wt) pairs and a source node.

Intuition:

Steps: 
1) Do a TopoSort (DFS/BFS). Let's do using DFS with Stack.
2) Create a distance array init with Inf, makr the distance[src]=0
3) Now for each elt in stack, pop one by one and explore the neighbors, update the distance if shorter found
4) Once stack is empty, distance array will store shortest distance from src node to each node in directed weighted graph. 

We don’t use Dijkstra for DAG because a DAG has no cycles, so we can process nodes in topologically sorted order and relax each edge only once. This gives O(V+E) and even supports negative weights, unlike Dijkstra which is slower and unnecessary here.



```python
class Solution:
    def shortestPathDirectedGraph(self, V:int, adj:List[List[int]], src:int)-> List[int]:
        # step1 do toposort using dfs and stack, assuming there's no cycle, so no need for detecting cycle, assume DAG given
        stack=[]
        visited=[0]*V
        def dfs(node):
            visited[node]=1
            for nei,wt in adj[node]:
                if not visited[nei]:
                    dfs(nei)
            stack.append(node)
        # for all components
        for i in range(V):
            if not visited[i]:
                dfs(i)
        
        
     
        # Step 2: Relax edges in topological order
        # Stack has the toposort, init dist with inf and distance[src]=0
        # directed weighted graph so adj list has node,wt
        distance=[float('inf')]*V
        distance[src]=0
        while stack:
            node=stack.pop()
            if distance[node] != float('inf'):  # process only reachable nodes
                for nei,wt in adj[node]:
                    if wt+distance[node]<distance[nei]:
                        distance[nei]=wt+distance[node]
        return distance


```


### Dijkstra's Algorithm

Time Complexity

**Using Queue**
I can do the Dijsktra's Algorithm using a simple queue. It works, but since we don't ahve the nodes or distacne sorted in the queue and we just follow a FIFO approach, we might do some extra operations and add and delete some redundant paths.
That's why it's preferred to use a PQ, where the dist is sorted, so we always get the minimum distance and node, and we update the dist for other ndoes based on this distance.

**Using Priority Queue**
Also we can do it using a simple queue, but it will be a bit more inefficient, so PQ is a better choice.

Steps for Dijkstra's:
We are finding shortest distance from src node to all nodes, so we init a dist array with inf and since shortest dist from start/src node to itself is 0, we do dist[src]=0 . We also use a PQ so we always greedily use the node with min dist to find shortes path. 
So init PQ with start node and 0 distance

Another important thing here is we don't use a visited array, we explore a single node mulitple times, i.,e explore all the paths to a node, and update its distacne when shortest distance found. However we use a greedy approach of always starting with a shortest distance path and node.

```python
class Solution:
    def Dijkstra(V:int, adj:List[List[int]], src:int)->List[int]:
        distance=[float('inf')]*V
        distance[src]=0
        pq=[(0,src)]
        # pq stores dist and node
        heapq.heapify(pq)
        while pq:
            dist,node=heapq.heappop(pq)
            # slight optimization here would be to ignore a path if the dist it offers is greater than our curr distance.
            # Why would I want to explore it , I want min dist to each node
            if dist>distance[node]:
                continue

            for nei,edg_wt in adj[node]:
                newDist=dist+edg_wt
                if newDist<distance[nei]:
                    distance[nei]=newDist
                    # this can be a possible shorter path for other nodes as well, so I explore all paths from this node to other nei with the dist
                    heapq.heappush(pq,(newDist,nei))
        return distance

```

Time Complexity for Dijkstra's Algorithm is (ElogV)

**Using Set**

Now how does set help us. If we observe, what does PQ do? It returns the shortest distance first from PQ. The set does that too right, it is made like that. It stores distance,node in sorted order, so that is done. Now where does it actually help. We update the distance when we find a shorter distacne right, but when we do that we add that ndoe and dist to pq/set in this case, so we can find other shorter paths. But what if we find  another shorter path than that, then the shorter path we added before becomes redundant, it doesnt serve any purpose. It wouldn't affect end result, but it sure will do some redundant/wasteful work. so if we find a shorter path first time , we just do the same process we did in pq, update dist and add it to the set. However, in case we find a shorter distance and the dist was not inf, i.e someone already came here and it was longer than us, we dont need it. Since this is set we cna remove, remember we cannot remove it from a pq. so we do set.remove(distance[currNode],currNode) and then update the distance[currNode]=newDist and add it to the set
The time complexity doesn't change much, but it is a bit better 

<!-- We dont have ordered set in python  -->



### Print Shortest Path (using Dijkstra's Algorithm)



## DP Dynamic Programming

### 1-D DP


Problem:

Intuition:

Steps:

Solution:

### 2-D DP or DP on Grids

Problem:

Intuition:

Steps:

Solution:

### DP On Subsequences with target

Technique for Subsequences, either you take an element, or you don't take it. That's the thumbrule. There may be modifications to picking depending on constraints. 

Problem:

Intuition:

Steps:

Solution:


### DP on Subsequences

Technique for Subsequences, either you take an element, or you don't take it. That's the thumbrule. There may be modifications to picking depending on constraints. 

Problem:

Intuition:

Steps:

Solution:

### Longest Common Subsequence

Problem: Given two strings text1 and text2, return the length of their longest common subsequence. If there is no common subsequence, return 0.

Intuition: Brute Force would be to generate all the subsequences for both the strings of len m and n, find the longest common in both these. This would be very inefficeint, as generating subsequences would be exponential (2^n and 2^m) and then comparing and finding the longest would also cost (2^n*2^m) accounting totaltime complexity as exponential (completely infeasible)

Steps: First Recursion--> Then Memoize--> Tabulation--> Space Optimisation if just using prev row or just prev indices, no need to carry entire dp matrix

Solution:
Since we are looking for common subsequence of two strings s1 and s2, if we start from end and compare each character, we have two options either characters match or they don't . 
If they match perfect we say common subsequence will be 1 +f(i-1,j-1) where i and j are indices that start from end of s1 and s2 respectively. 
Now in second case, if they dont match, what can we do, making sure that we are still havoing a valid subsequence, since we can skip/omit certain chars in a subsequence, we try omitting a char in s1 and s2 and chose the option that gives us the max common subsequence, i.e f(i-1,j) or f(i,j-1)


### Printing LCS



### Longest Palindromic Subsequence



### Longest Common Substring




### Minimum insertions to make string palindrome


### DP on Strings
Distinct Subsequences

Given two strings s and t, return the number of distinct subsequences of s which equals t.

Input: s = "rabbbit", t = "rabbit"
Output: 3
Explanation:
As shown below, there are 3 ways you can generate "rabbit" from s.
ra**bb**bit
ra**b**b**b**it
rab**bb**it

Intution:
What are we returning , nos/count of distinct subsequences, how do I find subsequences, find all using recursion, how do i make sure that it is valid and is equal to s2, well check char by char if they match it can be one of the valid if all chars match, right, also since you want to consider all the disticnt, you also need to consider other if you dont choose the char at that particular index, i.e consider the one that matches and also explore if you didn't choose that one. If they dont match, you really dont have any other choose other than moving to the next index. Since we want s2 inside of s1, if they dont match, just move s1 pointer, s2 pointer only moves if they match.

```python

```


### DP on Stocks


#### 122. Best Time to Buy and Sell Stock II

You are given an integer array prices where prices[i] is the price of a given stock on the ith day.

On each day, you may decide to buy and/or sell the stock. You can only hold at most one share of the stock at any time. However, you can sell and buy the stock multiple times on the same day, ensuring you never hold more than one share of the stock.

Find and return the maximum profit you can achieve.



Intuition:
Buy and then sell and then buy again. Cannot hold multiple stocks together, atmost one at a time
A lot of possible ways, and get best possible way from them.

So, we use a recursive function that tracks index (days) and buy(0,1). Now at each day if buy=1 (we can only sell or hold). so we return the max of these two options. However, if buy==0 then we can either buy, or do nothing. here also we return the max of these both. since we are startign from 0, our base case would be reahcing index n, at n, we just simply return 0, because we can do nothing at nth day.Now when we buy a stock we set buy=1 for future recursive calls, so they cannot buy stock unless the curr stock that is being held is sold off. 


```python

def maxProfit(self, prices: List[int]) -> int:

    def f(idx,buy):
        # base case
        if idx==n:
            return 0
        if dp[idx][buy]!=-1:
            return dp[idx][buy]
        if buy:
            # either sell or hold
            sell=prices[idx]+f(idx+1,0)
            hold=f(idx+1,1)
            dp[idx][buy]= max(sell,hold)
            return dp[idx][buy]
        else:
            buy=-prices[idx]+f(idx+1,1)
            nothing=f(idx+1,0)
            dp[idx][buy]=max(buy,nothing)
            return dp[idx][buy] 
    n=len(prices)
    dp=[[-1]*2 for _ in range(n)]
    return f(0,0)

```


```python

def maxProfit(self, prices: List[int]) -> int:
        # # Tabulation
        n=len(prices)
        ahead=[0,0]
        # base case and start from end 
        # if idx==n then return zero, so
        ahead[0]=ahead[1]=0
        cur=[0,0]
        # now start form back
        for idx in range(n-1,-1,-1):
            for buy in range(2):
                if buy:
                    profit=max((prices[idx]+ahead[0]), (ahead[1]))
                else:
                    profit=max(-prices[idx]+ahead[1],ahead[0])
                cur[buy]=profit
            ahead=cur
        return ahead[0]



```



#### Best Time to Buy and Sell Stock III

At max 2 transactions can be done. Other constraints remain same

<!-- # here the constraint is maximum times you can buy/sell is 2, so buy 2 stocks and then sell, return max profit 
# so we add another param that is cap. We start cap as 2, if cap==0, no more transaction can be done right, we updqate cap when we sell the stock, so when we sell 2 stocks our cap becomes 0 . this is one of the end condition.if cap==0 or we reach end of array we stop and return the profit earned . we finally return max profit on all possible ways
# if buy=0 0 means False cannot buy
# if buy==1, means go ahead buy  -->

```python
      def f(idx,buy,cap):
            # base case
            if idx==n or cap==0:
                return 0
            if dp[idx][buy][cap]!=-1:
                return dp[idx][buy][cap]
            if buy:
                # true cany buy or not buy
                # we dont update cap when we buy, we only do that when we sell
                purchase=-prices[idx]+f(idx+1,0,cap)
                nothing=f(idx+1,1,cap)
                dp[idx][buy][cap]=max(purchase,nothing)
            else:
                # buy=0 cannot buy, means either sell or hold it
                sell=prices[idx]+f(idx+1,1,cap-1)
                hold=f(idx+1,0,cap)
                dp[idx][buy][cap]=max(sell,hold)
            return dp[idx][buy][cap] 

        
        n=len(prices)
        # 3d dp array
        dp=[[[-1]*3 for _ in range(2)] for _ in range(n)]
        # we call 
        return f(0,1,2)

```

For above algo, we used nx2x3, there is one more variation where we can use nx4 dp array, here 4 refers to nos of total transactions. since we can only do 2 buy and 2 sell, we set dp=nx4 where 4 =2 buy + 2 sell. so f(idx,transaction) f(0,0) when even we can buy right, if odd that is we already bought so we can sell. Now when we sell or if we buy we incr transaction. Also whene transactions==4 we can break that means we did 2 buy and 2 sells

We will use this approach for next question to see how it works


#### Best Time to Buy and Sell Stock IV

This is same as prev. But here instead of 2 transactions, we are given k. so we first check if k>len(prices)
we reset it to len(prices), if less than that leave as is. Now just create dp array of size nx2xk and it will work 




## DP on LIS

### Longest Increasing Subsequence

Given a string, return the length of the longest increasing subsequence

Intuition:
We want longest subsequence which is increasing. so try all possible subsequences, now check which is increasing and longest, return that. This is the brute force solutio which is inefficient.

Can we do better? Yes, how about using recursion and then optimising it by storing intermediate results in a dp array
So, for recursion we try all the possible subsequences right? How? We either take an elt as part of the subsequence or not? How to know if we can pick the elt or not?

Let' start from the first index(0) and work our way towards the end of the given array. The only way to know if we can pick an elt is to know that the elt we are picking is strictly greater than the last elt we picked in our subsequence. Oh man!! How can we track that. So for this problem we will have i to track the indice of the array and j/prev_ind to track index of previous elt we picked in our subsequence. 

Let's start with f(0,-1), -1 here says we haven't picked anything yet
So the recurrence would look somehting like this for the pick case:
```code
    pick=0
    if nums[idx]>nums[prev_ind]:
    <!-- move to the next index and mark the curr index as prev_idx -->
        pick=1+f(i+1,idx)
    <!-- if we dont pick the elt, we simply add 0 to the length and let the prev_idx remain same as it was -->
    notpick=0+f(i+1,prev_idx)
```


```python
 def f(i,prev_idx):
            # base case
            if i==n:
                # end of array, cannot increase length, stop here
                return 0
            # pick only if no elt picked before or elt at prev_idx <elt at curridx     
            if dp[i][prev_idx+1]!=-1:
                return dp[i][prev_idx+1]
            pick=0
            if prev_idx==-1 or nums[i]>nums[prev_idx]:
                pick=1+f(i+1,i)
            notpick=0+f(i+1,prev_idx)
            dp[i][prev_idx+1]=max(pick,notpick)
            return dp[i][prev_idx+1] 
        n=len(nums)
        # in the dp I have to shift prev_index by 1, because it ranges from -1 to n-1, so we shift it by 1 index to the right to make it 0->n
        dp=[[-1]*(n+1) for _ in range(n)]
        return f(0,-1)
```
The above takes O(n^2) time and o(n^2) dp space + O(n) auxiliary space
We can bring down space complexity by using tabulation

```python
        # 1-D Tabulation
        # Idea here is to use 1d dp array that stores lenght of longest increasing subsequence at index
        # how can we figure that out if we are at an index, we will traverse  every elt until that index and see if it is smaller than our elt, if yes we update length =1+dp[prev] else we let it stay qas 1
        n=len(nums)
        dp=[1]*(n)
        maxi=0
        for i in range(n):
            for prev in range(i):
                if nums[prev]<nums[i] and dp[i]<1+dp[prev]:
                    dp[i]=1+dp[prev]
            maxi=max(maxi,dp[i])
        return maxi

```





### DP Imp Notes:
1) Always start with Recursion (if question revolves around trying all possibilities), steps for recursion:
    1) Express everything in terms of index (i,j) 
    2) Explore all possibilities, operations. What can we do at a particular index?
    3) Return count, max, min, print, whatever may be required
    4) Base case, utmost important, handle edge/boundary cases here

2) Try for Memoization by using dp array, steps to convert recursion to Memoization:
    1) Create a DP array of size of parameters, if only 1 param then 1d array else 2d
    2) Store the intermediate results and when a fucntion call is made first check if dp[i][j]!=-1 then return that result
    3) Finally once a funciton call is finished, store its result in dp[i][j]

3) Now go for Tabulation (Bottom-Up Approach)
    1) Create DP array of size of parameters
    2) Initialise base cases first, beacuse in bottom up approach, we work towards the solution using base cases, we build upon them unlike recursion which is top down where we start from any point and boil down to the base case and then bcaktrackeverything to final solution
    3) Now explore the remaining indices, most probably from 1->n or nested if 2 params.

For DP on Strings, remember to use shifting of indices when base cases tend to go -1, or negative. 





## Greedy

### Assign Cookies
Assume you are an awesome parent and want to give your children some cookies. But, you should give each child at most one cookie.

Each child i has a greed factor g[i], which is the minimum size of a cookie that the child will be content with; and each cookie j has a size s[j]. If s[j] >= g[i], we can assign the cookie j to the child i, and the child i will be content. Your goal is to maximize the number of your content children and output the maximum number.

Solution:
What we are trying to do, maximum nos of content children. If we think about solving this greedily, we can do that because if we sort the arrays, we have an idea that if a cookie cannot satisfy a child with greed factor j, it cannot satisfy the next child. so we try to use minimum size cookie that can make a child content

```python
def assignCookie( g:List[int], s:List[int])->int:
    g.sort()
    s.sort()
    i,j=0,0
    while i<len(g) and j<len(s):
        if s[j]>=g[i]:
            i+=1
            j+=1
        else:
            j+=1
    return i


```


### N meetings in one room

Problem:

Given one meeting room and N meetings represented by two arrays, start and end, where start[i] represents the start time of the ith meeting and end[i] represents the end time of the ith meeting, determine the maximum number of meetings that can be accommodated in the meeting room if only one meeting can be held at a time.


Intuition:
Since we want to maximise the number of meetings in one room, we want to take smaller meetings not longer ones, so we sort both the arrays together based on the end times, because if a meeting ends early , then only other meeting can be considered if its start time is after the end time of prev, if not ,we dont consider it.
We can increment count if meeting is considered and can also store the ids in a data structure, if we want the order

Time Complexity is O(nlogn) for sorting + O(n) for traversing the arrays. Space is O(1)


```python
class Solution:
    def maxMeetings(self, start, end):
        cnt=0
        freetime=0
        for st,end in sorted(zip(start,end),key= lambda x: x[1]):
            if st>freetime:
                cnt+=1
                freetime=end
        return cnt
```



### 435. Non-overlapping Intervals

Given an array of intervals intervals where intervals[i] = [starti, endi], return the minimum number of intervals you need to remove to make the rest of the intervals non-overlapping.

Note that intervals which only touch at a point are non-overlapping. For example, [1, 2] and [2, 3] are non-overlapping.

```python
class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        freetime=min(intervals, key=lambda x:x[0])[0]-1
        res=0
        for st,end in sorted(intervals,key=lambda x:x[1]):
            # print(st,end)
            if not st>=freetime:
                res+=1
            else:
                freetime=end
        return res


```


### 57. Insert Interval

You are given an array of non-overlapping intervals intervals where intervals[i] = [starti, endi] represent the start and the end of the ith interval and intervals is sorted in ascending order by starti. You are also given an interval newInterval = [start, end] that represents the start and end of another interval.

Insert newInterval into intervals such that intervals is still sorted in ascending order by starti and intervals still does not have any overlapping intervals (merge overlapping intervals if necessary).

Return intervals after the insertion.

Note that you don't need to modify intervals in-place. You can make a new array and return it.

```python
class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        # lets break it down into 3 segments, left part, overlapping part and right part, so we store the left intervals in our res until we find an overlap, now we merge this overlap, then again add remaining right intervals and finally return the res
        res=[]
        i=0
        n=len(intervals)
        # continue moving until newinterval's start overlaps or is beofre interval's end time
        while i<n and intervals[i][1]<newInterval[0]:
            res.append(intervals[i])
            i+=1
        # now overlapping part
        # continue until overlapp found, it will be until endtime of our newInterval is greater than start time of the next intervals
        
        while i<n and intervals[i][0]<=newInterval[1]:
            # i need to decide the start and end time fo this merged interval which will be min and max of these intervals 
            newInterval[0]=min(newInterval[0],intervals[i][0])
            newInterval[1]=max(newInterval[1],intervals[i][1])
            i+=1
        # now add it to the res
        res.append(newInterval)
        # now add reamining right side portion
        while i<n:
            res.append(intervals[i])
            i+=1
        return res
```
        

### 678. Valid Parenthesis String

Given a string s containing only three types of characters: '(', ')' and '*', return true if s is valid.
'*' could be treated as a single right parenthesis ')' or a single left parenthesis '(' or an empty string "".


```python
class Solution:
    def checkValidString(self, s: str) -> bool:

        # lets just think about this as a string with open and close brackets, no *, . lets maintain a cnt variable if open bracket, cnt+=1, if close cnt-=1. Now at the end if cnt==0 then open bracket==close bracket right? Only edge case is ordering of the brackets ))((, here cnt would first become -1,-2,-1,0 if we see cnt==0 at the end of string because left brackets==right brackets. But we want them to preserve the order, so to handle them we add a condition, if cnt<0, return False immediately, why? cnt only goes neagtive if right bracket's cnt surpasses left's cnt.

        # Now lets also add * to our problem
        # * can be (,),or ''. If using any of this in place of * gives us the correct string we return true, so we try all possibilities of this * i.e (, ), "" and see if we find a valid string
        

        # we start from index==0 and cnt==0, if any of the possibilities return True, we return True. 
        # when we encounter ( or ), we dont have much choice we just return whatever it returns. but for * we try all 3 possibilities
        def f(idx,cnt):
            # base case
            if cnt<0:
                return False
            if idx>=n:
                return cnt==0
            if dp[idx][cnt]!=-1:
                return dp[idx][cnt]
            # explore all ways
            if s[idx]=='(':
                dp[idx][cnt]=f(idx+1, cnt+1)
                return dp[idx][cnt] 
            if s[idx]==')':
                dp[idx][cnt]=f(idx+1, cnt-1)
                return dp[idx][cnt] 
            else:
                # * case, try all 3 possibilities, return true if any true
                # (,    ),    *
                dp[idx][cnt]=f(idx+1,cnt+1) or f(idx+1,cnt-1) or f(idx+1,cnt)
                return dp[idx][cnt]
        
        n=len(s)
        dp=[[-1]*n for _ in range(n)]
        return f(0,0)

        # this is recursion O(3^n), lets optimise it using memoization



        # Now dp still uses O(n^2) time and O(n^2) dp+ O(n) stack space
        # We can use tabulation and bring down space, but still O(n^2)



        # O(N) optimal solution
        mini,maxi=0,0
        for c in s:
            if c=='(':
                mini+=1
                maxi+=1
            elif c==')':
                mini-=1
                maxi-=1
            else:
                mini-=1
                maxi+=1
            if mini<0:
                mini=0
            if maxi<0:
                return False
        return mini==0



```








## Heaps

A Heap is a complete binary tree that satisfies the Heap Property. It is the most efficient way to implement a Priority Queue.

**Complete Binary Tree**: All levels are completely filled, except possibly the last level, which is filled from left to right. This allows for an array-based representation without pointers, making it space-efficient.

Heap Property:

Max-Heap: The value of every parent node is greater than or equal to the values of its children. The largest element is always at the root ($O(1)$ access).

Min-Heap: The value of every parent node is less than or equal to the values of its children. The smallest element is always at the root ($O(1)$ access).

Operation           Time Complexity,        Notes
Peek/Top,           O(1),                   Access the root (max/min element).
Insert,             O(logn),                Insertion at the end followed by sift-up (or bubble-up) to restore the heap property.
Extract (Max/Min),  O(logn),                Replace root with the last element, then use sift-down (or heapify) to restore the heap property.
Build Heap,         O(n),                   Building a heap from an array using a bottom-up approach is linear time.

**Whenever mention of Top-k largest, smallest , think of Heaps**



### Kth largest element in an Array

The "Top K"\K-th Element Pattern 

Goal: Find the Kth largest/smallest element, the Top K elements, or the K closest points.

Trick/Algorithm: 
Use a heap of size K.
To find the Kth Largest element (or the Top K elements), use a Min-Heap of size K. 
This keeps the K largest elements seen so far, with the overall smallest of this group (the $K^{th}$ largest) at the root.

To find the Kth Smallest element, use a Max-Heap of size K.Time Complexity: O(N log K). 
You iterate through $N$ elements, and each heap operation takes $O(log K) time. This is far better than O(N log N) sorting.


Root of Size-K Min-Heap The smallest among the $K$ largest elements $\implies$ the $K^{th}$ largest element.


```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        # optmised version using minheap of size k
        # if i have k elements in my heap, I'll always have kth largest element on my root
        minheap=[]
        for num in nums:
            if len(minheap)==k:
                # I need to check if number is greater than the number at root, I add it to the heap else discard it
                if minheap[0]<num:
                    heapq.heappop(minheap)
                    heapq.heappush(minheap,num)
            else:
                heapq.heappush(minheap,num)
        return minheap[0]

```
The total time complexity is $\mathbf{O(N \log K)}$. Since the heap size is capped at $K$, every insertion/deletion takes $O(\log K)$. This is a massive improvement over $O(N \log N)$ when $K \ll N$.


### Top-K frequent Elements


```python
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        # Since I need most freq elements, I'll use a Min heap of size k
        # first I need frequencies
        freq={}
        for num in nums:
            if num in freq:
                freq[num]+=1
            else:
                freq[num]=1
        # Now iterate over this freqmap and make sure you only store k elements at a time
        minheap=[]
        for num,cnt in freq.items():
            if len(minheap)==k:
                if cnt>minheap[0][0]:
                    heapq.heappop(minheap)
                    heapq.heappush(minheap,(cnt,num))
            else:
                heapq.heappush(minheap,(cnt,num))
        res=[val for cnt,val in minheap]
        return res
```

### 973. K Closest Points to Origin

Given an array of points where points[i] = [xi, yi] represents a point on the X-Y plane and an integer k, return the k closest points to the origin (0, 0).

The distance between two points on the X-Y plane is the Euclidean distance (i.e., √(x1 - x2)2 + (y1 - y2)2).

```python

class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        distances=[]
        # since closest points, we use Max heap of size k
        # first find distance for each point from the origin
        # store as tuple (dist,(x,y))
        maxheap=[]
        # since we don't have inbuilt maxheap , we negate and add
        for (x,y) in points:
            dist=math.sqrt((x)**2+(y)**2)
            # Now add the dist,coordinate pair to the maxheap
            if len(maxheap)==k:
                # heap is full, compare and then add
                # only add if smaller found
                # abs beacuse we negate while adding dist
                if dist<abs(maxheap[0][0]):
                    heapq.heappop(maxheap)
                    heapq.heappush(maxheap,(-dist,(x,y)))
                # else ignore this pair and move on
            else:
                # add directly
                heapq.heappush(maxheap,(-dist,(x,y)))
            
        # since just returning coordinates
        res=[[x,y] for dist,(x,y) in maxheap]
        return res

```







### Tries
Insert, search, startwith






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

### Observations for Solving problems

If subarray problems:
Brute Force is always to try out all possible subarrays and return those who align with constraints
If involves Subarray Sum, then use hashmap for optimization, use prefixsum

If involves product of subarray, then best bet is sliding window


Subset doesnot follow any order, the only constraint is elements should be from the array, it can be empty , elements, all the elements.
Subsequences on other hand needs to be ordered, no need to be contigous, but needs to be in order.
Substrings/subarray needs to be contigous as well as in order.



Hashmap Syntax :
```python
for k,v in sorted(freq.items(),key=lambda x:x[1], reverse=True):
    res+=v*(k)

for char in sorted(freq, key=freq.get, reverse=True):
    res+=(char)*freq[char]

```