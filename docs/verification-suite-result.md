# Verification Suite Result

## Test Case 1: Semantic Synonym Match
**Input:** data/tests/test1_input.json
```json
{
  "threshold": 0.78,
  "top_k": 2,
  "items": [
    {
      "id": "1",
      "text": "Network connection dropped because of server failure"
    },
    {
      "id": "2",
      "text": "Server error caused link disconnect"
    },
    {
      "id": "3",
      "text": "OutOfMemoryError: Java heap space exhausted"
    }
  ]
}
```
**Expected:** Items `1` and `2` should match, `3` shouldn't match. 

**Output:** 
```json
{
    "candidate_pairs": [
        {
            "left_id": "1", 
            "right_id": "2", 
            "similarity": 0.7929
        }
    ]
}
```
**Result:** ✅PASS

## Test Case 2: Zero Matches Under High Threshold
**Input:** data/tests/test2_input.json
```json
{
  "threshold": 0.98,
  "top_k": 5,
  "items": [
    {
      "id": "x",
      "text": "Database read timeout"
    },
    {
      "id": "y",
      "text": "Database connection lost"
    }
  ]
}
```
**Expected:** No items should match due to high similarity.

**Output:** 
```json
{
    "candidate_pairs": []
}
```
**Result:** ✅PASS

## Test Case 3: Empty and Boundary Inputs
**Input A:** data/tests/test3a_input.json
```json
{
  "threshold": 0.50,
  "top_k": 3,
  "items": []
}
```
**Expected:** Empty output due to the absence of items.

**Output:** 
```json
{
    "candidate_pairs": []
}
```
**Result:** ✅PASS

**Input B:** data/tests/test3b_input.json
```json
{
  "threshold": 0.50,
  "top_k": 3,
  "items": [
    { "id": "single", "text": "Generic API error" }
  ]
}
```
**Expected:** Empty output because of the presence of just one item.

**Output:** 
```json
{
    "candidate_pairs": []
}
```
**Result:** ✅PASS

## Test Case 4: Top-K Constraints
**Input:** data/tests/test4_input.json
```json
{
  "threshold": 0.60,
  "top_k": 1,
  "items": [
    {
      "id": "item-a",
      "text": "Database transaction timeout error"
    },
    {
      "id": "item-b",
      "text": "Database read timed out"
    },
    {
      "id": "item-c",
      "text": "Database write timed out"
    }
  ]
}
```
**Expected:** `item-a` will have similarity scores higher than the threshold with both `item-b` and `item-c`, but `top_k` should allow for `item-a` to match with only one item.

**Output:** 
```json
{
    "candidate_pairs": [
        {
            "left_id": "item-a", 
            "right_id": "item-c", 
            "similarity": 0.7846
        }
    ]
}
```
**Result:** ✅PASS

## Test Case 5: Bidirectional Deduplication
**Input:**data/tests/test5_input.json
```json
{
  "threshold": 0.70,
  "top_k": 5,
  "items": [
    { "id": "20", "text": "SSL Handshake failed on server" },
    { "id": "10", "text": "Server failed to establish SSL handshake" }
  ]
}
```
**Expected:** Output should contain only one match, `left_id` and `right_id` should be 10 and 20 respectively.

**Output:** 
```json
{
    "candidate_pairs": [
        {
            "left_id": "10", 
            "right_id": "20", 
            "similarity": 0.9416
        }
    ]
}
```
**Result:** ✅PASS

## Summary
All 5 must-pass test cases executed successfully with `bge-m3` as the embedding model.

---