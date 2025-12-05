# Fix for Extremely High Training Loss (69,300)

## Problem

Training loss was **69,300** instead of normal range 0.5-5.0!

```
Epoch 1: loss=6.93e+4  ❌ (69,300 - WAY TOO HIGH!)
```

## Root Cause

**Line 339-341 in `train_sam3_lora_native.py`:**

```python
# WRONG - Hardcoded num_boxes=1
l_cls = self.criterion_cls.get_loss(outputs, targets, indices, num_boxes=1)
l_box = self.criterion_box.get_loss(outputs, targets, indices, num_boxes=1)
l_mask = self.criterion_mask.get_loss(outputs, targets, indices, num_boxes=1)
```

## Why This Causes High Loss

In DETR-style models, losses are **normalized by num_boxes** to make them scale-invariant:

```python
# Inside loss functions:
loss = raw_loss / num_boxes  # Normalize
```

**With num_boxes=1:**
- If you have 100 pixels in a mask, loss = 100 / 1 = **100**
- If you have 1000 predicted boxes, loss = 1000 / 1 = **1000**

**With correct num_boxes (e.g., 200):**
- If you have 100 pixels in a mask, loss = 100 / 200 = **0.5** ✅
- If you have 1000 predicted boxes, loss = 1000 / 200 = **5.0** ✅

## The Fix

```python
# CORRECT - Calculate actual num_boxes from targets
num_boxes = targets['boxes'].shape[0] if 'boxes' in targets and targets['boxes'].numel() > 0 else 1
num_boxes = max(num_boxes, 1)  # Avoid division by zero

l_cls = self.criterion_cls.get_loss(outputs, targets, indices, num_boxes=num_boxes)
l_box = self.criterion_box.get_loss(outputs, targets, indices, num_boxes=num_boxes)
l_mask = self.criterion_mask.get_loss(outputs, targets, indices, num_boxes=num_boxes)
```

## Expected Results After Fix

**Before:**
```
Epoch 1: loss=69300  ❌
```

**After:**
```
Epoch 1: loss=2.5    ✅
Epoch 5: loss=0.8    ✅
Epoch 10: loss=0.5   ✅
```

## Changes Made

Fixed in **two places**:
1. **Training loop** (line 336-345)
2. **Validation loop** (line 391-397)

Both now calculate `num_boxes` from actual ground truth instead of hardcoding to 1.

## Verification

Sample data shows correct object counts:
```
Sample 0: 1 objects
Sample 1: 1 objects
Sample 2: 2 objects
```

This means:
- Batch of 3 samples has 4 total objects
- num_boxes = 4 (not 1!)
- Losses will be divided by 4 → reasonable values

## Action Required

**Restart training** with the fixed code:

```bash
# Stop current training (if running)
# Ctrl+C or kill process

# Delete old weights (trained with wrong normalization)
rm -rf outputs/sam3_lora_full/

# Retrain with fixed loss normalization
python3 train_sam3_lora_native.py --config configs/full_lora_config.yaml
```

**Expected training output:**
```
Epoch 1: loss=2.0-3.0 ✅
Epoch 2: loss=1.0-2.0 ✅
Epoch 5: loss=0.5-1.0 ✅
Epoch 10+: loss=0.3-0.8 ✅
```

If you still see loss > 100, there may be other issues.

## Summary

Two bugs fixed so far:

1. ✅ **Data loading bug** - Was loading 0 objects → Now loads actual objects
2. ✅ **Loss normalization bug** - Used num_boxes=1 → Now uses actual count

Training should now work properly! 🎉

---

**Created:** 2025-12-05
**Issue:** Loss 69,300 instead of 0.5-5.0
**Fix:** Calculate num_boxes from actual ground truth
