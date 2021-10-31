Original Reference (UC Berkeley CS61a Fall 2021):
https://cs61a.org/articles/debugging/#common-bugs
- [Error Types](#error-types)
  - [1. SyntaxError](#1-syntaxerror)
  - [2. IndentationError](#2-indentationerror)
  - [3. TypeError](#3-typeerror)
  - [4. NameError](#4-nameerror)
  - [5. IndexError](#5-indexerror)
- [Common Bugs](#common-bugs)
  - [1. Spelling](#1-spelling)
  - [3. Missing close quotes](#3-missing-close-quotes)
  - [4. = vs. ==](#4--vs-)
  - [5. Infinite Loops](#5-infinite-loops)
  - [6. Off-by-one errors](#6-off-by-one-errors)
# Error Types
## 1. SyntaxError
**Cause**: code syntax mistake

**Example**:

```python
  File "file name", line number
    def incorrect(f)
                    ^
SyntaxError: invalid syntax
```
**Solution**: the ^ symbol points to the code that contains invalid syntax. The error message doesn't tell you what is wrong, but it does tell you where.
**Notes**: Python will check for SyntaxErrors before executing any code. This is different from other errors, which are only raised during runtime.

## 2. IndentationError

**Cause**: improper indentation

**Example**:
```python
  File "file name", line number
    print('improper indentation')
IndentationError: unindent does not match any outer indentation level
```
**Solution**: The line that is improperly indented is displayed. Simply re-indent it.
**Notes**: If you are inconsistent with tabs and spaces, Python will raise one of these. Make sure you use spaces! (It's just less of a headache in general in Python to use spaces and all cs61a content uses spaces).

## 3. TypeError
**Cause 1**:

- Invalid operand types for primitive operators. You are probably trying to add/subract/multiply/divide incompatible types.
- **Example**:
```python
TypeError: unsupported operand type(s) for +: 'function' and 'int'
```

**Cause 2**:

- Using non-function objects in function calls.
- Example:
```python
>>> square = 3
>>> square(3)
Traceback (most recent call last):
  ...
TypeError: 'int' object is not callable
```
**Cause 3**:

- Passing an incorrect number of arguments to a function.
- **Example**:
```python
>>> add(3)
Traceback (most recent call last):
  ...
TypeError: add expected 2 arguments, got 1
```

## 4. NameError

**Cause**: variable not assigned to anything OR it doesn't exist. This includes function names.

**Example**:

```python
File "file name", line number
  y = x + 3
NameError: global name 'x' is not defined
```
**Solution**: Make sure you are initializing the variable (i.e. assigning the variable to a value) before you use it.
**Notes**: The reason the error message says "global name" is because Python will start searching for the variable from a function's local frame. If the variable is not found there, Python will keep searching the parent frames until it reaches the global frame. If it still can't find the variable, Python raises the error.

## 5. IndexError
**Cause**: trying to index a sequence (e.g. a tuple, list, string) with a number that exceeds the size of the sequence.

**Example**:

```python
File "file name", line number
  x[100]
IndexError: tuple index out of range
```

**Solution**: Make sure the index is within the bounds of the sequence. If you're using a variable as an index (e.g. seq[x], make sure the variable is assigned to a proper index.

# Common Bugs
## 1. Spelling
Python is case sensitive. The variable hello is not the same as Hello or hello or helo. This will usually show up as a NameError, but sometimes misspelled variables will actually have been defined. In that case, it can be difficult to find errors, and it is never gratifying to discover it's just a spelling mistake.

##2. Missing Parentheses
A common bug is to leave off the closing parenthesis. This will show up as a SyntaxError. Consider the following code:
```python
def fun():
    return foo(bar()   # missing a parenthesis here

fun()
```
Python will raise a SyntaxError, but will point to the line after the missing parenthesis:
```python
File "file name", line "number"
    fun()
      ^
SyntaxError: invalid syntax
```
In general, if Python points a SyntaxError to a seemingly correct line, you are probably forgetting a parenthesis somewhere.

## 3. Missing close quotes
This is similar to the previous bug, but much easier to catch. Python will actually tell you the line that is missing the quote:
```python
File "file name", line "number"
  return 'hi
           ^
SyntaxError: EOL while scanning string literal
EOL stands for "End of Line."
```
## 4. = vs. ==
The single equal sign = is used for assignment; the double equal sign == is used for testing equivalence. The most common error of this form is something like:

```python
if x = 3:
```

## 5. Infinite Loops
Infinite loops are often caused by while loops whose conditions never change. For example:
```python
i = 0
while i < 10:
    print(i)
```
Sometimes you might have incremented the wrong counter:
```python
i, n = 0, 0
while i < 10:
    print(i)
    n += 1
```

## 6. Off-by-one errors
Sometimes a while loop or recursive function might stop one iteration too short. Here, it's best to walk through the iteration/recursion to see what number the loop stops at.