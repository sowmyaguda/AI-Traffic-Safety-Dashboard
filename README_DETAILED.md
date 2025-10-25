# CV Analytics Dashboard - Backend Documentation

## What Is This?

This is an **AI-powered traffic safety analytics system** that helps you understand vehicle behavior and predict where crashes are likely to happen. Think of it as having a smart assistant that can analyze thousands of vehicle records and tell you exactly where, when, and why crashes occur.

The magic? You just talk to it in plain English. No coding, no complex menus - just ask questions like you're talking to a person.

## What Can It Do?

Imagine you're a traffic engineer with 100,000 vehicle records. Normally, you'd spend hours in Excel trying to find patterns. This system does it in seconds and tells you things you'd never notice on your own.

### The Cool Stuff (AI Features)

**1. Talk to Your Data**
- Ask: "Show me speeding on I-70" - it shows you instantly
- Ask: "When are crashes most likely?" - it analyzes patterns and tells you
- Ask: "Find crash hotspots" - it identifies danger zones with GPS coordinates

**2. Crash Hotspot Clustering**
- Uses AI to find where crashes cluster together
- Shows you red circles on the map marking danger zones
- Tells you: "70% of crashes happen in just 5 locations"
- Gives you exact GPS coordinates for intervention

**3. Braking Intensity Analysis**
- Classifies every hard braking event: Emergency, Hard, or Moderate
- Shows which roads have the most extreme braking
- Discovers correlation: "Roads with emergency braking also have more crashes"
- Color-codes the map: Dark Red = Emergency, Red = Hard, Orange = Moderate

**4. Crash Prediction (The Really Smart Stuff)**
- Finds hard braking near past crash locations (near-misses)
- Identifies roads with high braking but NO crashes yet → future crash zones!
- Calculates risk scores: "This road scores 25/100 - CRITICAL risk"

**5. When Will It Happen? (Temporal Analysis)**
- Discovers: "Crashes peak at 12 AM, 1 AM, 11 AM, 1 PM, 3 PM, 6 PM"
- Shows you: "Hard braking predicts crashes 15 hours in advance"
- Automatically filters the map to dangerous hours

**6. How Safe Is This Road?**
- Scores every road 0-100 (higher = safer)
- Combines: crash history + braking patterns + speed variance
- Rankings: Critical (<30), High (30-60), Moderate (60-80), Low (80-100)

## How Does It Work?

### The Simple Explanation

1. **You upload data**: Vehicle locations, speeds, timestamps
2. **AI analyzes**: Finds patterns humans would miss
3. **You ask questions**: In plain English
4. **System responds**: With insights, charts, and maps

### The Technical (But Still Simple) Explanation

**GPU-Powered Speed**
- Uses RAPIDS cuDF instead of regular pandas
- Processes 100,000 records in milliseconds instead of seconds
- Your laptop's graphics card does the heavy lifting

**Smart Natural Language Understanding**
- You type: "Which roads are most dangerous?"
- AI understands you mean: Calculate safety scores and rank roads
- Runs the analysis automatically
- Shows you results with charts and maps

**Real-Time Everything**
- Every command updates the map instantly
- Charts appear in under a second
- No waiting, no page refreshes

## What's Inside? (File Breakdown)

### The Main File: `app.py` (180KB)

This is the entire system. Everything lives here:

**Lines 1-500**: Setup and imports
- Loads libraries (Streamlit, cuDF, Folium for maps, Plotly for charts)
- Initializes session state (remembers what you've done)

**Lines 500-700**: Natural Language Understanding
- Connects to LLM (Groq, OpenAI, or Gemini)
- Parses your questions into commands
- Examples: "Show crashes" → `action: "show_crashes"`

**Lines 700-1100**: Data Processing
- Loads vehicle and crash data
- Filters by road, time, speed
- Converts GPS coordinates, calculates speeds

**Lines 1100-1600**: Map Generation
- Creates interactive Folium maps
- Color-codes markers (speed, braking severity)
- Auto-zooms to filtered data
- Adds crash hotspot circles
- Shows popups with details

**Lines 1600-2100**: AI Analysis Functions

1. **`analyze_proximity_risk()`** (Lines 1645-1770)
   - Finds hard braking within 200m of crashes
   - Calculates correlation rates
   - Identifies future crash zones

2. **`calculate_road_safety_scores()`** (Lines 1772-1840)
   - Scores roads 0-100
   - Formula: 40% crashes + 40% braking + 20% speed variance

3. **`analyze_temporal_risk()`** (Lines 1840-2070)
   - Hour-by-hour crash patterns
   - Leading indicator analysis (how far in advance can we predict?)
   - Day-of-week patterns

4. **`analyze_crash_hotspots()`** (Lines 1960-2060)
   - DBSCAN clustering algorithm
   - Groups crashes within 50m radius
   - Returns cluster centers and sizes

5. **`analyze_braking_intensity()`** (Lines 2100-2200)
   - Classifies: Emergency (≤-0.7g), Hard (≤-0.5g), Moderate (≤-0.3g)
   - Calculates intensity scores per road
   - Correlates with crash data

**Lines 2200-2700**: Visualization Functions
- Chart generators for every analysis
- Bar charts, scatter plots, pie charts, histograms
- Temporal pattern visualizations
- All use Plotly for interactivity

**Lines 2700-3800**: Command Execution
- Handles every user command
- Routes to the right analysis function
- Generates response text
- Updates the map and charts

**Lines 3800-4200**: User Interface
- Sidebar: API key input, data upload
- Main area: Map + Chat interface
- Analytics section: All the charts

### Support Files

**`run.sh`** - Startup Script
```bash
#!/bin/bash
# Activates RAPIDS environment
# Starts Streamlit on port 8501
# Displays startup messages
```

**`requirements.txt`** - Python Dependencies
```
streamlit
streamlit-folium
folium
plotly
cudf-cu12  # GPU-accelerated dataframes
scikit-learn  # For clustering
```

**`api.py`** - Optional REST API
- If you want to call the system from other apps
- Same features, JSON responses instead of UI

## How to Use It

### First Time Setup

1. **Install RAPIDS Environment**
```bash
conda create -n rapids-25.10 -c rapidsai -c conda-forge -c nvidia \
    rapids=25.10 python=3.11 cudatoolkit=12.0
```

2. **Activate Environment**
```bash
conda activate rapids-25.10
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Get API Key** (Free!)
- Go to: https://console.groq.com/
- Sign up (takes 2 minutes)
- Create API key
- Copy it

### Starting the System

**Simple Way:**
```bash
./run.sh
```

**Manual Way:**
```bash
streamlit run app.py --server.port 8501 --server.address localhost --server.baseUrlPath /cv-dashboard
```

Then open: http://localhost:8501/cv-dashboard

### Using the System

1. **Enter API Key** (left sidebar)
   - Paste your Groq API key
   - Select model (llama-3.3-70b-versatile recommended)

2. **Upload Data** (left sidebar)
   - Vehicle CSV: Must have columns: latitude, longitude, speed, timestamp, road
   - Crash CSV: Must have columns: latitude, longitude, timestamp, RoadName

3. **Ask Questions!** (main chat box)

**Starter Questions:**
```
Show all data
Filter by I-70
Show crashes
Color by speed
```

**Analysis Questions:**
```
Find crash hotspots
Analyze braking intensity
When are crashes most likely?
Where will crashes happen next?
Calculate road safety scores
```

**Advanced Questions:**
```
Show hard braking on I-70 from 7am to 9am
Which roads have emergency braking?
Show me the top 5 most dangerous spots
What time of day is most dangerous?
```

## Everything You Can Do (Complete Operations Guide)

Here's EVERY operation you can perform, in plain English, the way you'd actually ask:

### Basic Map & Data Operations

**See Everything:**
- "Show all data" - Shows all 100,000 vehicles on the map
- "Show me everything" - Same thing, different wording
- "Display all vehicles" - The AI gets it
- "Reset" - Clears all filters, shows everything again

**Filter by Location (Roads):**
- "Filter by I-70" - Show only vehicles on I-70
- "Show MO 141" - Works with any road name
- "Show me vehicles on Main Street" - Any road works
- "Filter by I-70 and MO 141" - Multiple roads at once
- "Which roads do I have data for?" - Lists all available roads

**Filter by Time:**
- "Show data from 7am to 9am" - Rush hour morning
- "Show afternoon traffic" - It understands "afternoon"
- "Filter by 2pm to 5pm" - Specific hours
- "Show data between 08:00 and 10:00" - 24-hour format works too
- "What happened at midnight?" - Specific time

**Filter by Speed:**
- "Show speeding" - Shows vehicles going over the speed limit
- "Filter by speed over 80" - Custom speed threshold
- "Show me cars going faster than 70 mph" - Works
- "Which vehicles are speeding on I-70?" - Combines location + speed

**Filter by Vehicle Behavior:**
- "Show hard braking" - Any deceleration ≤ -0.3g
- "Show emergency braking" - Really hard stops (≤ -0.7g)
- "Filter by deceleration below -0.5" - Custom threshold
- "Show me all the hard braking events on I-70" - Combines filters

### Crash Analysis

**View Crashes:**
- "Show crashes" - Displays all crash locations
- "Show me the accidents" - Same thing
- "Display crash data" - Works
- "Where did crashes happen?" - AI understands the question

**Filter Crashes:**
- "Show crashes on I-70" - Crashes on a specific road
- "Show crashes from 7am to 9am" - Time-based
- "Show crashes in the last hour" - Relative time (if your data supports it)

**Crash Analysis:**
- "Analyze crash-braking correlation" - Finds near-misses (hard braking within 200m of crashes)
- "Show proximity risk" - Same analysis, different wording
- "Are crashes related to hard braking?" - AI figures out what you mean

### AI-Powered Predictions (The Smart Stuff!)

**1. Crash Hotspot Clustering**

What it does: Finds where crashes cluster together in tiny areas

Ask me:
- "Find crash hotspots" - Identifies all danger zones
- "Where are crashes clustered?" - Same thing
- "Show me crash clusters" - Works
- "Identify danger zones" - AI gets it

What you get back:
- Red circles on the map showing hotspot locations
- Exact GPS coordinates for each hotspot
- How many crashes in each cluster
- "Top 3 hotspots contain 68% of all crashes"

**2. Braking Intensity Analysis**

What it does: Classifies every hard braking event by severity

Ask me:
- "Analyze braking intensity" - Full analysis
- "Show braking severity" - Same thing
- "Which roads have the hardest braking?" - AI figures it out
- "Classify braking events" - Works

What you get back:
- Emergency: ≤-0.7g (dark red on map)
- Hard: ≤-0.5g (red on map)
- Moderate: ≤-0.3g (orange on map)
- Chart showing distribution
- "I-70 has 45 emergency braking events"
- Correlation with crashes

**3. Crash Prediction (Where Will Crashes Happen Next?)**

What it does: Identifies future crash zones based on hard braking patterns

Ask me:
- "Where will crashes happen next?" - Full predictive analysis
- "Predict future crashes" - Same thing
- "Find high-risk roads with no crashes yet" - More specific
- "Show me near-miss events" - Focuses on that aspect

What you get back:
- Roads with hard braking but NO crashes yet
- Risk scores for each location
- "Route X scores 25/100 - CRITICAL risk"
- GPS coordinates for intervention
- Leading indicators (how far in advance we can predict)

**4. Temporal Risk Analysis (When Are Crashes Most Likely?)**

What it does: Hour-by-hour and day-of-day crash patterns

Ask me:
- "When are crashes most likely?" - Full temporal analysis
- "What time of day is most dangerous?" - Same thing
- "Show crash patterns by hour" - More specific
- "Which days have the most crashes?" - Day-of-week analysis

What you get back:
- Peak crash hours: "12 AM, 1 AM, 11 AM, 1 PM, 3 PM, 6 PM"
- Charts showing crash distribution by hour
- Day-of-week patterns
- "Hard braking predicts crashes 15 hours in advance"
- Time-lag analysis

**5. Road Safety Scoring (How Safe Is Each Road?)**

What it does: Scores every road 0-100 (higher = safer)

Ask me:
- "Calculate road safety scores" - Full analysis
- "Which roads are most dangerous?" - Gets the top results
- "Rate road safety" - Works
- "Show me safety rankings" - Same thing

What you get back:
- Every road scored 0-100
- Rankings: Critical (<30), High (30-60), Moderate (60-80), Low (80-100)
- "I-70: 28/100 CRITICAL"
- "MO 141: 65/100 MODERATE"
- Formula: 40% crashes + 40% braking + 20% speed variance

**6. Comprehensive AI Insights (Everything at Once)**

What it does: Runs ALL analyses and gives you a complete report

Ask me:
- "AI insights" - Full comprehensive report
- "Analyze everything" - Same thing
- "Give me the full analysis" - Works
- "Show all insights" - AI understands

What you get back:
- Crash hotspots
- Braking intensity
- Temporal patterns
- Road safety scores
- Proximity risk analysis
- Crash prediction
- Complete summary with GPS coordinates
- Actionable recommendations

### Visualization & Charts

**Generate Charts:**
- "Show speed distribution" - Histogram of speeds
- "Chart speeds by road" - Bar chart
- "Show crash timeline" - Crashes over time
- "Visualize braking patterns" - Various braking charts

**Map Display Options:**
- "Color by speed" - Gradient from slow (blue) to fast (red)
- "Color by braking" - Shows braking intensity colors
- "Show heat map" - Density visualization
- "Zoom to I-70" - Auto-zooms to filtered data

### Data Management

**Upload Data:**
- Use sidebar to upload Vehicle CSV and Crash CSV
- App auto-detects columns
- Converts AccMagnitude to deceleration_g automatically

**Check Status:**
- "How much data do I have?" - Shows record counts
- "List available roads" - Shows all roads in dataset
- "What time range is covered?" - Shows data span

### Combining Operations (The Power Move!)

You can chain operations together! Each step filters further:

**Example Workflow 1: Morning Rush Hour Safety Check**
```
Step 1: "Filter by I-70"
Step 2: "Show data from 7am to 9am"
Step 3: "Show hard braking"
Step 4: "Analyze braking intensity"
```

**Example Workflow 2: Crash Investigation**
```
Step 1: "Show crashes on MO 141"
Step 2: "Show hard braking on MO 141 from 2pm to 5pm"
Step 3: "Analyze crash-braking correlation"
```

**Example Workflow 3: Safety Report for Presentation**
```
Step 1: "Show all data"
Step 2: "Find crash hotspots"
Step 3: "Calculate road safety scores"
Step 4: "When are crashes most likely?"
Step 5: "AI insights" (pulls it all together)
```

### Pro Tips

**Natural Language = You Can Ask Anything:**
- The AI understands variations, so don't memorize exact phrases
- "Show me" = "Display" = "Filter by" = "I want to see"
- "Crashes" = "Accidents" = "Collisions"
- "Hard braking" = "Sudden stops" = "Heavy deceleration"

**Mix and Match:**
- Combine road + time + behavior: "Show hard braking on I-70 from 7am to 9am"
- Stack filters: Start broad, then narrow down with each question

**Reset When Stuck:**
- Type "Show all data" or "Reset" to clear all filters
- Useful if you're not sure what's currently filtered

**Speed vs Performance:**
- 100,000 records load in <1 second (with GPU)
- Complex AI analyses take 1-3 seconds
- Map rendering takes 2-4 seconds
- Worth the wait!

**Charts Stack:**
- Each analysis adds charts below the map
- Scroll down to see all visualizations
- Click X button to close charts you don't need

### What You'll Get Back

Every operation responds in plain English, like:

**After filtering:**
"Showing 8,542 vehicles on I-70. Zoomed map to data extent."

**After crash analysis:**
"Found 15 crash locations on I-70. 3 crashes occurred during morning rush hour."

**After AI analysis:**
"AI has discovered critical safety insights:
- 5 crash hotspots identified (top 3 = 68% of crashes)
- 42 high-risk roads with no crashes yet
- Peak crash hours: 12 AM, 1 AM, 11 AM, 1 PM, 3 PM
- Hard braking predicts crashes 15 hours in advance"

**After safety scoring:**
"Road Safety Analysis:
- I-70: 28/100 CRITICAL
- MO 141: 65/100 MODERATE
- Main St: 82/100 LOW RISK"

### Common Questions (Just Ask!)

**"What roads do I have?"**
"Available roads: I-70, MO 141, MO 30, Main Street, Highway 61... (lists all)"

**"Show me the most dangerous spots"**
→ Runs crash hotspot analysis

**"Why are there so many crashes on I-70?"**
→ Runs comprehensive analysis on I-70 only

**"Which road should we focus on first?"**
→ Calculates safety scores, shows worst first

**"Is morning or evening more dangerous?"**
→ Runs temporal analysis

This is your complete operations guide. Remember: just talk to it like you're asking a traffic safety expert for help. The AI figures out what you mean!

## The Data Pipeline

### Input Data Format

**Vehicle Data (CSV or Parquet):**
```csv
latitude,longitude,speed,timestamp,road,speed_limit
38.5394,-90.2707,65.5,2025-07-28 08:15:23,I-70,60
```

**Optional columns** (makes it better):
- VehicleID, VehicleType, Bearing
- AccMagnitude (acceleration - we'll calculate if missing)

**Crash Data (CSV):**
```csv
latitude,longitude,timestamp,RoadName,severity
38.5401,-90.2712,2025-07-28 09:30:00,I-70,High
```

### What Happens Inside

1. **Data Loading** (Lines 440-490)
   - Checks for pre-loaded parquet file
   - Converts column names to standard format
   - Creates `deceleration_g` from acceleration data

2. **Smart Calculations**
   - **Deceleration**: Change in speed over time → g-force units
   - **Hard Braking**: Any deceleration ≤ -0.3g
   - **Emergency**: ≤ -0.7g (very dangerous)

3. **Spatial Analysis**
   - Converts distances: meters → degrees (111km = 1 degree)
   - Proximity check: Within 200m = 200/111000 degrees
   - Clustering: DBSCAN with 50m radius

4. **Temporal Analysis**
   - Extracts hour from timestamp
   - Groups by hour/day
   - Calculates time gaps between events

## AI Integration

### How Natural Language Works

**Your Input:** "Show me crash hotspots"

**What Happens:**
1. Your text → LLM (Groq/OpenAI/Gemini)
2. LLM returns: `{"action": "crash_hotspots", "params": {}}`
3. System calls: `analyze_crash_hotspots()`
4. Results appear on map + charts

**The Prompt (Lines 515-596):**
```
You are a command parser for a vehicle analytics dashboard...

Examples:
"Find crash hotspots" -> {"action": "crash_hotspots", "params": {}}
"Show speeding on I-70" -> {"action": "filter_road", "params": {"road": "I-70", "show_speeding": true}}
```

### Supported LLM Providers

**Groq (Recommended - Fast & Free)**
- Models: llama-3.3-70b, mixtral-8x7b, gemma-7b
- Speed: ~500 tokens/second
- Cost: Free tier available

**OpenAI**
- Model: gpt-4o-mini
- Speed: ~100 tokens/second
- Cost: Pay per use

**Google Gemini**
- Model: gemini-1.5-flash
- Speed: ~200 tokens/second
- Cost: Free tier available

## The Math Behind It

### Safety Score Formula
```
Safety Score (0-100) = 100 - [
  (crash_rate / max_crash_rate × 40) +
  (braking_rate / max_braking_rate × 40) +
  (speed_variance / max_variance × 20)
]
```

**Why this works:**
- Higher crashes = lower score (40% weight)
- More hard braking = lower score (40% weight)
- More speed variance = lower score (20% weight)
- Result: 0 = worst, 100 = safest

### Intensity Score Formula
```
Intensity Score = (Emergency × 3) + (Hard × 2) + (Moderate × 1)
```

**Why this works:**
- Emergency braking is 3× more important than moderate
- Hard braking is 2× more important
- Captures severity, not just frequency

### Clustering Algorithm (DBSCAN)

**What it does:**
- Groups crashes that are within 50m of each other
- Finds "dense" areas of crashes
- Ignores isolated incidents

**Parameters:**
- `eps = 50m / 111000` (radius in degrees)
- `min_samples = 2` (minimum crashes to form a cluster)

**Why DBSCAN:**
- Doesn't need to know number of clusters in advance
- Handles irregular shapes (not just circles)
- Ignores noise (isolated crashes)

## Performance Specs

**With RAPIDS cuDF (GPU):**
- 100,000 records loaded: <1 second
- Filter operation: <100ms
- Complex analysis: 1-3 seconds
- Map rendering: 2-4 seconds

**Without GPU (pandas fallback):**
- 100,000 records loaded: 3-5 seconds
- Filter operation: 500ms-1s
- Complex analysis: 5-15 seconds
- Map rendering: same

**Memory Usage:**
- Base app: ~500MB
- With 100K records: ~800MB
- With charts rendered: ~1.2GB

## Troubleshooting

### "Column(s) ['deceleration_g'] do not exist"

**Solution:** The app now auto-fixes this!
- Uses existing `AccMagnitude` column
- Converts to `deceleration_g` automatically
- Just refresh your browser

### "Connection error" / "No API key"

**Solution:**
1. Get free API key from https://console.groq.com/
2. Enter in left sidebar
3. Try command again

### Map not updating

**Solution:**
- App uses smart caching
- Updates when data actually changes
- If stuck, ask: "Show all data" to reset

### Charts not appearing

**Solution:**
- Scroll down! Charts render below the map
- Click the X button to close old charts
- Charts stack vertically

## What Makes This Special

### 1. No Learning Curve
- Talk to it like a human
- No SQL, no code, no manuals
- If you can type a question, you can use it

### 2. Instant Insights
- GPU acceleration = millisecond responses
- Real-time map updates
- No waiting for reports

### 3. Predictive, Not Just Descriptive
- Doesn't just show what happened
- Predicts where crashes WILL happen
- Identifies early warning signs

### 4. Comprehensive Analysis
- 8 different AI-powered features
- 15+ chart types
- 4 simultaneous map overlays
- All in one system

### 5. Built for Presentations
- Professional visualizations
- Auto-generates insights
- GPS coordinates ready for action
- One-click comprehensive reports

## Example Use Cases

### Morning Traffic Operations
```
> Show all data
[Map loads with 100,000 vehicles]

> Filter by I-70
[Zooms to I-70, shows traffic]

> Show hard braking from 7am to 9am
[654 hard braking events highlighted]

> When are crashes most likely?
[Analysis shows: 11 AM, 1 PM, 3 PM peak hours]
```

### Safety Engineering Report
```
> Find crash hotspots
[Identifies 15 hotspots, top 3 = 68% of crashes]

> Calculate road safety scores
[I-70: 28/100 CRITICAL, MO 141: 65/100 MODERATE]

> Where will crashes happen next?
[12 high-risk roads identified with no crashes yet]

> AI insights
[Comprehensive report with all analyses]
```

### Incident Investigation
```
> Show crashes on MO 141
[8 crashes displayed]

> Show hard braking on MO 141 from 2pm to 5pm
[23 hard braking events in that window]

> Analyze crash-braking correlation
[Found 5 near-miss events, 62% correlation rate]
```

## Future Enhancements (Ideas)

- **Weather integration**: Correlate rain/snow with crashes
- **Real-time data**: Stream live vehicle data
- **Mobile app**: Field use for traffic officers
- **PDF reports**: Auto-generate safety reports
- **Multi-city**: Compare cities, share best practices

## Technical Stack

**Core:**
- Python 3.11
- Streamlit (web framework)
- RAPIDS cuDF (GPU dataframes)

**Data Processing:**
- Pandas (CPU fallback)
- NumPy (numerical operations)
- Scikit-learn (clustering)

**Visualization:**
- Folium (interactive maps)
- Plotly Express (interactive charts)
- Streamlit-Folium (map integration)

**AI/LLM:**
- Groq API
- OpenAI API
- Google Gemini API

**Performance:**
- CUDA (GPU acceleration)
- Session state caching
- Lazy chart rendering

## Credits & License

**Built for:** Traffic safety analysis and crash prediction

**Created:** October 2025

**Powered by:**
- RAPIDS.AI for GPU acceleration
- Streamlit for the interface
- Groq for natural language understanding

**License:** Use it, modify it, improve it - just make roads safer!

---

## Need Help?

**Quick Start:** Run `./run.sh` and open http://localhost:8501/cv-dashboard

**Questions?** The system literally answers questions - just ask it!

**Problems?** Check the troubleshooting section above.

**Want to contribute?** The entire system is in `app.py` - it's all there!

---

Remember: This isn't just data visualization. This is an AI assistant that understands traffic safety and can predict crashes before they happen.
