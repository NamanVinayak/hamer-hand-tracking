# EGOCENTRIC DATA AGENCY - MASTER REFERENCE DOCUMENT
**Last Updated:** December 22, 2025  
**Owner:** Naman Vinayak

---

## 1. BUSINESS OVERVIEW

### What We Do
We collect and process **egocentric (first-person POV) video data** of humans performing real-world tasks, then sell annotated datasets to robotics/AI companies training humanoid robots and foundation models.

### The Opportunity
- **Market Size:** $6B+ invested in robotics in first 7 months of 2025
- **Q1 2025 alone:** $2.26B into robotics startups (70%+ to warehouse/industrial)
- **Key Insight:** Robotics companies need proprietary datasets + AI software bundled together

### Our Advantage
**"India Arbitrage"** - Film in Western-style environments (5-star hotels, modern facilities) in India at 1/5 US costs
- Data collection: $10-15/hour (India) vs $50-100/hour (US)
- Same quality, Western brands/environments
- Zero domain gap for US/Canadian robot deployment

---

## 2. MARKET VALIDATION

### Companies Recently Funded

**Sensei Robotics (YC S24)**
- "Scale AI for robotics data"
- YC-backed ($500K minimum)
- Marketplace model: contractors collect demonstration data
- Pricing: $30/hour for accepted footage

**Cortex AI (YC F25)** - DIRECT COMPETITOR
- Launched Fall 2025 (3 months ago)
- Building "world's largest egocentric + robot dataset"
- Founder: Lucas Ngoo (ex-CTO of Carousell, $1B+ marketplace)
- Backed by: YC + Pioneer Fund
- **What they deliver:**
  - Egocentric video
  - Hand/body pose tracking
  - Depth data
  - Subtask labels
  - Marketplace where workplaces get paid to host data collection

**Apple EgoDex (May 2025)** - FREE COMPETITION
- 829 hours of egocentric manipulation video
- Hand tracking (25 joints per hand)
- 194 household tasks
- **Completely free and open-source**
- Collected with Apple Vision Pro

### What This Means
✅ Market is validated (Sensei/Cortex funded)  
✅ There IS demand for this data  
⚠️ We're late to market (Cortex launched 3 months ago, Apple released free dataset 7 months ago)  
⚠️ Need to differentiate or find niche

---

## 3. CURRENT MVP: HOSPITALITY DATASET

### Product
**"Hospitality Pilot Pack"** - 100 hours of footage across 20+ hotel rooms in India
- Bed-making, cleaning, restocking, organizing
- Multiple room types (luxury to budget)
- Western-style environments
- 4K/60fps egocentric capture

### Why Hospitality?
- Broader market appeal than textiles
- Aligns with what humanoid robot companies want
- Easy to film in India (abundant hotels)
- Repeatable tasks with natural variance

---

## 4. TECHNICAL STACK (POST-PROCESSING)

### What Robotics Companies Need

| Component | Required? | What It Is |
|-----------|-----------|------------|
| Raw Video | ✅ CRITICAL | 4K/60fps egocentric footage |
| Hand Tracking | ✅ CRITICAL | 3D coordinates of hand joints |
| Object Masks | ✅ CRITICAL | Pixel-level object identification |
| Task Labels | ✅ CRITICAL | "making bed", "folding towel" |
| Depth Data | ⚠️ NICE TO HAVE | 3D spatial information |
| Body Pose | ⚠️ NICE TO HAVE | Upper body skeleton |

**Decision:** Focus on the 4 CRITICAL components first. Add depth/body pose later if customers demand it.

---

### STAGE 1: HAND TRACKING 🔥 CRITICAL

**Technology:** MediaPipe Hands (Google, free, open-source)

**What it does:**
- Detects and tracks 21 hand landmarks per hand in 3D
- Works on regular video (no special hardware needed)
- Runs on GPU (RunPod)

**Implementation:**
```python
import mediapipe as mp
hands = mp.solutions.hands

# Process video frame by frame
results = hands.process(frame)
# Returns: 21 hand landmarks (x,y,z coordinates)
```

**Output:** JSON file with hand positions per frame
```json
{
  "frame_0001": {
    "left_hand": {
      "wrist": [x, y, z],
      "thumb_tip": [x, y, z],
      "index_tip": [x, y, z]
      // ... 21 points total
    },
    "right_hand": { ... }
  }
}
```

**Vibe-Codable:** ✅ 90%  
**Why This Matters:** Hand trajectories are 50% of the value. Robots need to learn how humans move their hands to manipulate objects.

---

### STAGE 2: OBJECT SEGMENTATION 🔥 CRITICAL

**Technology:** Meta SAM 3 (Segment Anything Model 3, Nov 2025 release, free)

**What it does:**
- Automatically identifies and segments objects in video
- Text prompts: "segment all hands", "segment all towels"
- Tracks objects across video frames
- 8.4x faster than manual annotation

**Implementation:**
```python
from sam3 import SAM3Model
model = SAM3Model()

# Segment objects with text prompts
masks = model.segment_video(
    "hotel_cleaning.mp4",
    text_prompt="hands, towel, pillow, bed, sheet"
)
```

**Output:** PNG mask files showing which pixels are which objects
```
masks/
├── frame_0001_hands.png
├── frame_0001_towel.png
├── frame_0001_bed.png
└── ...
```

**Vibe-Codable:** ✅ 90%  
**Why This Matters:** Robots need to know "I'm holding a towel" vs "I'm holding a plate". Object identification is 30% of the value.

---

### STAGE 3: TASK ANNOTATION 🔥 CRITICAL

**Technology:** Simple Streamlit web interface

**What it does:**
- Human annotator watches video
- Labels what task is happening
- Marks start/end timestamps

**Implementation:**
```python
import streamlit as st

st.video("hotel_001.mp4")
task = st.text_input("Task name: (e.g., making_bed)")
start_time = st.slider("Start time", 0, 300)
end_time = st.slider("End time", 0, 300)

if st.button("Save Annotation"):
    save_to_csv(video_id, task, start_time, end_time)
```

**Output:** CSV file with task labels
```csv
video_id,task,start_time,end_time
hotel_001,making_bed,0:05,1:23
hotel_001,folding_towel,1:24,2:45
hotel_001,restocking_minibar,2:46,3:30
```

**Vibe-Codable:** ✅ 100%  
**Why This Matters:** Robots need to know "this is a bed-making task" to learn the right behavior. Task labels are 15% of the value.

**Cost:** $5-10 per hour of video (annotator time in India)

---

### STAGE 4: QUALITY CONTROL

**Technology:** Streamlit dashboard

**What it does:**
- Human reviewer checks processed videos
- Validates hand tracking looks correct
- Validates object masks are accurate
- Approves or flags for re-processing

**Implementation:**
```python
# Show video with hand tracking overlaid
st.video_with_overlays(video, hand_data, masks)

# QC checkboxes
st.checkbox("Hand tracking accurate?")
st.checkbox("Object masks correct?")
st.checkbox("Task labels match video?")

if st.button("Approve for Delivery"):
    mark_as_ready(video_id)
```

**Vibe-Codable:** ✅ 90%  
**Why This Matters:** Cannot deliver bad data to paying customers

---

## 5. WHAT WE DELIVER TO CUSTOMERS

### Package Structure
```
dataset_package_hotel_001/
│
├── videos/
│   └── hotel_001.mp4                    # Original 4K egocentric video
│
├── hand_tracking/
│   └── hotel_001_hands.json             # 21 hand landmarks per frame
│
├── object_masks/
│   ├── frame_0001_hands.png
│   ├── frame_0001_towel.png
│   ├── frame_0001_bed.png
│   └── ... (one PNG per object per frame)
│
├── task_labels.csv                       # Task annotations with timestamps
│
└── metadata.json                         # Camera info, resolution, fps, etc
```

### Data Format Details

**Hand Tracking JSON:**
- 30 fps (matches video)
- 21 landmarks per hand (MediaPipe standard)
- 3D coordinates (x, y, z) in normalized space
- Confidence scores per landmark

**Object Masks:**
- PNG format (same resolution as video)
- Binary masks (255 = object, 0 = background)
- One file per object per frame
- Can be loaded into any ML framework

**Task Labels CSV:**
- Simple format: video_id, task, start_time, end_time
- Human-readable task names
- Timestamps in HH:MM:SS format

---

## 6. ECONOMICS

### Cost Per 100 Hours of Video

**Shooting (Week 1):**
- 10 workers @ $5/hour × 10 hours = $500
- iPhone rentals/mounts = $200
- **Subtotal: $700**

**Processing (Week 2-3):**
- RunPod GPU compute (100 hours video) = $300
- 2 annotators @ $5/hour × 20 hours = $200
- **Subtotal: $500**

**QC & Delivery (Week 4):**
- QC reviewer @ $5/hour × 20 hours = $100
- Cloud storage/bandwidth = $50
- **Subtotal: $150**

**Total Cost:** ~$1,350 per 100 hours

### Pricing
- **Low end:** $30/hour (Sensei rate) = $3,000
- **Mid range:** $40-50/hour = $4,000-5,000
- **High end (exclusive):** $75/hour = $7,500

**Profit Margin:** $1,650 - $6,150 per 100 hours (55-82%)

---

## 7. COMPETITION ANALYSIS

### Direct Competitors

**Cortex AI (YC F25)** ⚠️ BIGGEST THREAT
- **Launched:** Fall 2025 (3 months ago)
- **Founder:** Lucas Ngoo (successful $1B+ marketplace founder)
- **Backing:** YC + institutional investors
- **Model:** Marketplace (workplaces get paid to host collection)
- **Delivers:** Egocentric video + hand/body pose + depth + subtask labels
- **Advantage:** Better pedigree, institutional backing, marketplace model
- **Our Edge:** India arbitrage (1/5 cost), focused on hospitality niche

**Sensei Robotics (YC S24)**
- **Model:** Marketplace with contractors using low-cost hardware
- **Pricing:** $30/hour for accepted footage
- **Scale:** Plans for "thousands of operators worldwide"
- **Risk:** Acceptance criteria unclear, payment only for accepted footage
- **Our Edge:** Professional cinematography quality, predictable output

**Apple EgoDex (Released May 2025)** ⚠️ FREE COMPETITION
- **Size:** 829 hours, completely free and open-source
- **Quality:** Vision Pro hardware capture (very high quality)
- **Coverage:** 194 household tasks
- **Threat:** Why pay for data when Apple gave away 829 hours free?
- **Our Edge:** Customers need MORE data (829 hours isn't enough), specific tasks, ongoing collection

### Indirect Competition
- **In-house data teams:** Tesla ($48/hour), Figure AI, Physical Intelligence
- **Scale AI:** General data labeling (not specialized in robotics)
- **Individual contractors:** Gig workers on Sensei/Cortex platforms

---

## 8. OUR GAPS VS TOP COMPETITORS

| Feature | Apple EgoDex | Cortex AI | Us (MVP) |
|---------|--------------|-----------|----------|
| Video quality | ✅ Excellent | ✅ Good | ✅ Good |
| Hand tracking | ✅ 25 joints | ✅ Full body | ✅ 21 joints |
| Object masks | ✅ | ✅ | ✅ |
| Task labels | ✅ | ✅ | ✅ |
| **Depth data** | ✅ | ✅ | ❌ |
| **Body pose** | ✅ | ✅ | ❌ |
| Scale | ✅ 829 hours | ⚠️ Unknown | ⚠️ Starting |
| Cost | FREE | ⚠️ Unknown | $30-75/hour |

**The Honest Assessment:**
- We're competitive on video/hand/object/task
- Missing depth and body pose (can add later if needed)
- Biggest gap: Brand recognition and scale

---

## 9. IS THIS FUNDABLE?

### ✅ Why It COULD Be Funded
- Market validated ($2.26B in Q1 2025 to robotics)
- Two YC companies (Sensei, Cortex) prove business model works
- India arbitrage is real competitive advantage
- "Picks and shovels" business (infrastructure)

### ❌ Why It's RISKY
- Apple released 829 hours FREE 7 months ago
- Cortex launched 3 months ago with better pedigree
- We're "just video production" without tech layer
- Market may consolidate around 1-2 winners quickly

### The Reality
**Not fundable YET because:**
- No customers
- No tech built
- No traction
- Unclear moat (what stops Cortex from doing India too?)

**COULD be fundable IF:**
- Build the tech (annotation pipeline)
- Get 3-5 paying customers
- Prove unit economics
- Show we can deliver better quality or lower cost than Cortex

**Better Path:** Bootstrap for 3-6 months, prove model works, THEN raise if needed

---

## 10. TECHNOLOGY HIRING (IF NEEDED)

### If You Go the Tech Route (Not Bootstrapping)

**Critical Hire #1: Computer Vision Engineer**
- **Skills:** Python, PyTorch, video processing, MediaPipe, SAM integration
- **Cost:** $80-120K/year (India) or $150-200K/year (US/Canada)
- **Role:** Build the annotation pipeline (Stages 1-4)
- **When:** After you have 3-5 customers saying "I'll pay for this"

**Critical Hire #2: ML Ops Engineer**
- **Skills:** Cloud infrastructure, data pipelines, API design, Docker
- **Cost:** Similar to above
- **Role:** Make data deliverable, queryable, scalable
- **When:** After first engineer is productive (Month 4-6)

**You (Naman) Stay As:**
- **Operations:** Managing data collection in India
- **Strategy:** Business development, customer relationships
- **Quality:** Ensuring video quality meets professional standards

---

## 11. BUILD TIMELINE

### Month 1: Hand Tracking Validation
**Goal:** Prove hand tracking works well enough

**Tasks:**
- Set up MediaPipe on RunPod (1 week)
- Process 10 sample videos (1 week)
- Validate accuracy is acceptable (1 week)
- Document the workflow (1 week)

**Deliverable:** Working hand tracking pipeline

**Can Vibe-Code:** ✅ YES (90%)

---

### Month 2: Add Segmentation + Annotation
**Goal:** Complete the core pipeline

**Tasks:**
- Integrate SAM 3 for object segmentation (1 week)
- Build Streamlit annotation interface (1 week)
- Train 2 annotators in India (1 week)
- Process 20 hours of fully annotated data (1 week)

**Deliverable:** 20 hours of fully processed dataset

**Can Vibe-Code:** ✅ YES (85%)

---

### Month 3: Quality Control + Customer Validation
**Goal:** Production-ready dataset + early customers

**Tasks:**
- Build QC dashboard (1 week)
- Process 100 hours of data (2 weeks)
- Show to 5 potential customers (1 week)

**Deliverable:** 100-hour production dataset + customer feedback

**Can Vibe-Code:** ✅ YES (85%)

---

### Month 4-6: Scale or Pivot
**Decision Point:**

**If 3+ customers say "yes, I'll pay":**
→ Consider raising funding to scale
→ Hire computer vision engineer
→ Build production infrastructure

**If 0-2 customers say "maybe":**
→ Pivot to become supplier for Cortex/Sensei marketplace
→ Or find a specific niche they don't serve
→ Or explore different task categories

---

## 12. IMMEDIATE NEXT STEPS

### This Week (Dec 22-29, 2025)
**Priority 1:** Validate hand tracking works

**Action:**
1. Get 1 sample video (5-10 minutes of hotel task)
2. Use Cursor/Claude to write MediaPipe script
3. Run it on RunPod
4. Visualize the output
5. Check if hand tracking quality is good enough

**Time:** 1 day if focused  
**Cost:** ~$5 (RunPod)

**Success Criteria:** Hand tracking follows hands accurately 80%+ of the time

---

### Next 30 Days (Jan 2026)
**Priority 1:** Build complete pipeline MVP

**Action:**
1. Complete Stage 1: Hand tracking working
2. Complete Stage 2: SAM 3 integration working
3. Complete Stage 3: Simple annotation tool
4. Process 10 hours of video end-to-end

**Success Criteria:** Can deliver a complete package (video + hands + masks + labels)

---

### Next 90 Days (Q1 2026)
**Priority 1:** Customer validation

**Action:**
1. Process 100 hours of hospitality data
2. Show to 10 robotics companies
3. Get 3-5 to say "yes, we'll pay for this"
4. Decide: raise funding vs bootstrap vs pivot

**Success Criteria:** Clear signal if this is a real business or not

---

## 13. KEY METRICS TO TRACK

### Production Metrics
- Hours of video collected per week
- Cost per hour of video (target: <$15)
- Processing time per hour of video
- Hand tracking accuracy %
- Object segmentation accuracy %

### Business Metrics
- Number of leads contacted
- Conversion rate (leads → customers)
- Average sale price per hour
- Customer acquisition cost
- Gross margin %

### Quality Metrics
- QC rejection rate (target: <5%)
- Customer satisfaction score
- Re-work rate

---

## 14. RISKS & MITIGATION

### Risk 1: Cortex Dominates Market
**Probability:** High  
**Impact:** Fatal  
**Mitigation:** Find niche they don't serve (specific industries, specific tasks) or become supplier to them

### Risk 2: Not Enough Customer Demand
**Probability:** Medium  
**Impact:** Fatal  
**Mitigation:** Validate with customers BEFORE building full tech stack

### Risk 3: Quality Not Good Enough
**Probability:** Low  
**Impact:** High  
**Mitigation:** Test with real customers early, iterate based on feedback

### Risk 4: Can't Compete on Price
**Probability:** Medium  
**Impact:** High  
**Mitigation:** India arbitrage gives 5x cost advantage, focus on quality over lowest price

### Risk 5: Apple's Free Dataset Kills Market
**Probability:** Low  
**Impact:** Medium  
**Mitigation:** 829 hours isn't enough for training, customers need ongoing data collection

---

## 15. DECISION POINTS

### Should I Build This?
**YES, IF:**
- You can vibe-code 70%+ yourself (you can)
- You can validate customers want this in 90 days
- You're OK with bootstrap for 6+ months
- You have access to India operations (you do)

**NO, IF:**
- You need funding immediately
- You can't commit 3-6 months
- You're not comfortable with technical work
- You can't access shooting locations in India

### Should I Raise Funding?
**NOT YET. Build MVP first.**

**Raise ONLY AFTER:**
- 3-5 paying customers
- Proven unit economics
- Clear differentiation from Cortex
- Validated the market wants YOUR data

---

## 16. OPEN QUESTIONS

### Customer Questions
- [ ] Do customers prefer hospitality tasks or other categories?
- [ ] What's the acceptable price range per hour?
- [ ] Is depth data actually required or nice-to-have?
- [ ] What file formats do they prefer?

### Technical Questions
- [ ] Is 85% hand tracking accuracy good enough?
- [ ] Do we need body pose tracking?
- [ ] How important is multi-angle capture?
- [ ] Should we offer depth estimation in post?

### Business Questions
- [ ] Should we partner with Cortex/Sensei as supplier?
- [ ] Should we focus on one vertical (hospitality) or diversify?
- [ ] Should we offer exclusive vs non-exclusive licenses?
- [ ] What's our unique selling proposition vs Cortex?

---

## 17. CONTACT INFO & REFERENCES

### Key Tools
- **MediaPipe Hands:** https://google.github.io/mediapipe/solutions/hands
- **Meta SAM 3:** https://ai.meta.com/sam2/ and https://huggingface.co/facebook/sam3
- **RunPod:** https://www.runpod.io/
- **Streamlit:** https://streamlit.io/

### Competitor Links
- **Cortex AI:** https://cortexrobot.ai/
- **Sensei Robotics:** https://senseirobotics.com/
- **Apple EgoDex:** https://machinelearning.apple.com/research/egodex-learning-dexterous-manipulation

### Current Leads (from previous conversations)
- **Sureform (Ananth Kashyap):** ananth@sureformhq.com - YC W25, data company
- **Physical Intelligence (Lerrel Pinto):** Meeting had Nov 2025
- **1X Technologies (Xiaolong Wang):** Meeting had Nov 2025

---

## 18. VERSION HISTORY

**v1.0 - Dec 22, 2025**
- Initial master document
- Based on research and conversations through Dec 22, 2025
- Tech stack defined (no depth, focus on hand tracking + segmentation + labels)
- Timeline and economics validated
- Competition analyzed (Cortex, Sensei, Apple)

---

## SUMMARY: THE ONE-PAGE VERSION

**What:** Egocentric data collection company for robotics AI training  
**How:** Film hotel tasks in India, process with MediaPipe + SAM 3, sell to robotics companies  
**Advantage:** 1/5 cost of US collection, professional cinematography quality  
**Competition:** Cortex AI (direct), Sensei (marketplace), Apple (free 829 hours)  
**Tech:** MediaPipe (hands) + SAM 3 (objects) + Human (labels) = Complete dataset  
**Gap vs Leaders:** No depth data, no body pose (both can be added later)  
**Economics:** ~$1,350 cost per 100 hours, sell at $3,000-7,500 = 55-82% margin  
**Timeline:** 3 months to MVP, validate customers in 90 days  
**Risk:** Late to market, Cortex has head start, Apple gave away free data  
**Fundable?** Not yet. Build MVP first, get customers, then consider funding  
**Next Step:** Validate hand tracking works this week with 1 sample video  

---

*END OF MASTER DOCUMENT*