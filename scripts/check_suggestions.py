import json
from pathlib import Path
from collections import Counter, defaultdict

suggestions_dir = Path("data/suggestions_v3")
suggestion_files = list(suggestions_dir.glob("*_suggestions.json"))

all_types = []
per_file_counts = defaultdict(lambda: defaultdict(int))

for sfile in suggestion_files:
    with open(sfile) as f:
        suggestions = json.load(f)
    
    for s in suggestions:
        etype = s['evidence_type']
        all_types.append(etype)
        per_file_counts[sfile.name][etype] += 1

print(f"="*60)
print(f"FINAL PRE-ANNOTATION STATISTICS")
print(f"="*60)
print(f"Total files processed: {len(suggestion_files)}")
print(f"Total suggestions: {len(all_types)}")
print()

# Overall distribution
counts = Counter(all_types)
print("Overall Distribution:")
for etype in ['outer_boundary', 'pattern_region', 'unclear_region']:
    count = counts[etype]
    pct = (count / len(all_types)) * 100 if all_types else 0
    print(f"  {etype}: {count:4d} ({pct:5.1f}%)")

print()

# Per-image averages
avg_outer = sum(c['outer_boundary'] for c in per_file_counts.values()) / len(suggestion_files)
avg_pattern = sum(c['pattern_region'] for c in per_file_counts.values()) / len(suggestion_files)
avg_unclear = sum(c['unclear_region'] for c in per_file_counts.values()) / len(suggestion_files)

print("Average per image:")
print(f"  outer_boundary: {avg_outer:.1f}")
print(f"  pattern_region: {avg_pattern:.1f}")
print(f"  unclear_region: {avg_unclear:.1f}")
print()

# Sanity checks
print(f"="*60)
print("SANITY CHECKS:")
print(f"="*60)

if counts['outer_boundary'] == len(suggestion_files):
    print("✓ Exactly 1 outer_boundary per image (PERFECT!)")
elif counts['outer_boundary'] < len(suggestion_files) * 0.9:
    print(f"⚠️  Some images missing outer_boundary ({counts['outer_boundary']}/{len(suggestion_files)})")
elif counts['outer_boundary'] > len(suggestion_files) * 1.1:
    print(f"⚠️  Too many outer_boundary ({counts['outer_boundary']}/{len(suggestion_files)})")
else:
    print(f"✓ outer_boundary count looks good ({counts['outer_boundary']} for {len(suggestion_files)} images)")

if counts['pattern_region'] > len(suggestion_files) * 10:
    print(f"✓ Good pattern_region coverage (~{avg_pattern:.0f} per image)")
elif counts['pattern_region'] > len(suggestion_files):
    print(f"⚠️  Moderate pattern_region coverage (~{avg_pattern:.0f} per image)")
else:
    print(f"⚠️  Low pattern_region coverage (~{avg_pattern:.0f} per image)")

if 5 <= avg_unclear <= 20:
    print(f"✓ Reasonable unclear_region count (~{avg_unclear:.0f} per image)")
else:
    print(f"⚠️  Unclear_region: {avg_unclear:.0f} per image (might be high/low)")

print()
print(f"="*60)
print(f"READY FOR MANUAL REVIEW!")
print(f"="*60)
print(f"Next step: python scripts/04_review_suggestions.py")
print(f"="*60)