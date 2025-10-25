# AI-Powered Traffic Safety Dashboard

## What Is This?

This is a smart system that helps you understand vehicle behavior and predict where crashes are likely to happen. The cool part? You just talk to it in plain English. No coding, no complex menus - just ask questions like you're talking to a person.

Think of it as having a traffic safety expert who can analyze 100,000 vehicle records in seconds and tell you exactly where, when, and why crashes occur.

## What Makes It Special?

**You Can Talk to Your Data:**
- Ask: "Show me speeding on I-70" - it shows you instantly
- Ask: "When are crashes most likely?" - it analyzes patterns and tells you
- Ask: "Find crash hotspots" - it identifies danger zones with GPS coordinates

**It Predicts Crashes Before They Happen:**
- Finds hard braking events near past crash locations (near-misses)
- Identifies roads with hard braking but NO crashes yet - these are future crash zones
- Calculates risk scores for every road
- Tells you: "This road scores 25/100 - CRITICAL risk, intervene now"

**It's Fast:**
- Uses your computer's graphics card (GPU) to process data
- 100,000 vehicle records load in under 1 second
- Analysis completes in 1-3 seconds
- No waiting around

## What Can It Do?

### 1. Find Crash Hotspots
Groups crashes that happen in the same spot and shows you:
- Exact GPS coordinates
- How many crashes in each hotspot
- "70% of crashes happen in just 5 locations"

### 2. Analyze Braking Intensity
Classifies every hard braking event:
- Emergency braking (≤-0.7g) - shown in dark red
- Hard braking (≤-0.5g) - shown in red
- Moderate braking (≤-0.3g) - shown in orange
- Shows which roads have the most extreme braking

### 3. Predict Future Crashes
This is the really smart part:
- Finds hard braking near past crash locations
- Identifies roads with high braking but no crashes yet
- These are your future crash zones
- Gives you risk scores so you know where to focus

### 4. Tell You When Crashes Happen
Hour-by-hour analysis:
- "Crashes peak at 12 AM, 1 AM, 11 AM, 1 PM, 3 PM, 6 PM"
- "Hard braking predicts crashes 15 hours in advance"
- Shows you dangerous times automatically

### 5. Score Road Safety
Rates every road from 0-100 (higher = safer):
- Combines crash history + braking patterns + speed variance
- I-70: 28/100 CRITICAL - needs immediate attention
- MO 141: 65/100 MODERATE - monitor closely
- Highway 61: 82/100 LOW RISK - doing well

### 6. Natural Language Interface
The magic ingredient:
- No SQL queries to write
- No programming needed
- Just type questions in plain English
- The AI figures out what you mean

## Demo Videos

See the system in action! Watch these quick demonstrations:

### Video 1: Data Filtering & Map Operations
**[Watch Demo →](https://mailmissouri-my.sharepoint.com/:v:/g/personal/sghmy_umsystem_edu/EYsDPrMNdphFmLeUAeEIl2UB62uXshg7xhVrtNOFOTLjXA)**

Learn how to:
- Load vehicle and crash data
- Filter by road, time, and speed
- Navigate the interactive map
- Use natural language commands

### Video 2: Plotting & Visualizations
**[Watch Demo →](https://mailmissouri-my.sharepoint.com/:v:/g/personal/sghmy_umsystem_edu/ET_N0XtdQwtMlwpurQ1BThoB9ho8EOXqqqn9APgtvdeYAQ)**

Explore the visualization features:
- Speed distribution charts
- Crash pattern analysis
- Road-by-road comparisons
- Interactive plotting capabilities

### Video 3: AI-Powered Analysis
**[Watch Demo →](https://mailmissouri-my.sharepoint.com/:v:/g/personal/sghmy_umsystem_edu/EeMNR7DXsM5FnNds-JyZzocBf7EGsTWYvqNUuWRuAAVThw)**

See the AI features in action:
- Crash hotspot clustering
- Braking intensity analysis
- Road safety scoring
- Predictive crash analytics
- Comprehensive insights generation

**Tip:** These demos show real-world usage scenarios. Follow along to get started quickly!

---

## How to Get Started

### What You Need

- A computer with an NVIDIA graphics card (recommended, but not required)
- Python 3.11
- About 10 minutes to set up
- A free API key from Groq (takes 2 minutes to get)

### Installation Steps

**Step 1: Create the Environment**
```bash
conda create -n rapids-25.10 -c rapidsai -c conda-forge -c nvidia \
    rapids=25.10 python=3.11 cudatoolkit=12.0
```

**Step 2: Activate It**
```bash
conda activate rapids-25.10
```

**Step 3: Install Dependencies**
```bash
cd interactive-ai-dashboard/backend
pip install -r requirements.txt
```

**Step 4: Get Your Free API Key**
- Go to: https://console.groq.com/
- Sign up (takes 2 minutes)
- Create an API key
- Copy it

**Step 5: Start the System**
```bash
./run.sh
```

That's it! Open your browser to: http://localhost:8501/cv-dashboard

## How to Use It

### First Time Setup

1. **Enter your API key** in the left sidebar
2. **Upload your data files**:
   - Vehicle CSV: needs latitude, longitude, speed, timestamp, road
   - Crash CSV: needs latitude, longitude, timestamp, RoadName
3. **Start asking questions** in the chat box

### Example Questions to Try

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

### Real-World Example

Let's say you're a traffic engineer checking morning rush hour safety:

```
You: Show all data
System: [Displays 100,000 vehicles on map]

You: Filter by I-70
System: [Zooms to I-70, showing 8,542 vehicles]

You: Show hard braking from 7am to 9am
System: [Highlights 654 hard braking events]

You: Analyze braking intensity
System: [Shows 45 emergency braking events, 203 hard, 406 moderate]
        "Emergency braking correlation with crashes: 78%"

You: Where will crashes happen next?
System: "3 high-risk zones identified:
         - Mile 223: 87% crash probability
         - Clayton Rd intersection: 12 near-miss events
         - Recommend immediate intervention"
```

## What Data Do You Need?

### Vehicle Data (CSV or Parquet)

Your file should have these columns:
```csv
latitude,longitude,speed,timestamp,road,speed_limit
38.5394,-90.2707,65.5,2025-07-28 08:15:23,I-70,60
```

**Required:**
- latitude, longitude - GPS coordinates
- speed - vehicle speed in MPH
- timestamp - when the reading was taken
- road - which road the vehicle is on

**Optional (but helpful):**
- speed_limit - posted speed limit
- AccMagnitude - acceleration data (we calculate if missing)
- VehicleID, VehicleType, Bearing

### Crash Data (CSV)

```csv
latitude,longitude,timestamp,RoadName,severity
38.5401,-90.2712,2025-07-28 09:30:00,I-70,High
```

**Required:**
- latitude, longitude - where the crash happened
- timestamp - when it happened
- RoadName - which road

## How It Works Behind the Scenes

### The AI Part

When you type a question:
1. Your text goes to an AI language model (Groq, OpenAI, or Gemini)
2. The AI figures out what you want: "They want to see crash hotspots"
3. The system runs the analysis
4. Results appear on the map with charts

### The Smart Analysis

**Crash Prediction:**
- Looks at every hard braking event (when someone slams on brakes)
- Checks if it's near a past crash location (within 200 meters)
- Finds roads with lots of hard braking but no crashes yet
- These roads are danger zones waiting to happen
- Gives each road a risk score

**Safety Scoring Formula:**
```
Safety Score (0-100) = 100 - [
  (crash_rate × 40%) +
  (braking_rate × 40%) +
  (speed_variance × 20%)
]
```

- More crashes = lower score
- More hard braking = lower score
- More speed variation = lower score
- Result: 0 = most dangerous, 100 = safest

**Clustering (Finding Hotspots):**
- Groups crashes that are within 50 meters of each other
- Finds areas where crashes cluster together
- Ignores isolated one-off crashes
- Shows you the exact GPS coordinates

### The Speed

**With a Graphics Card (GPU):**
- 100,000 records load in under 1 second
- Filtering takes less than 100 milliseconds
- Complex analysis takes 1-3 seconds
- Really, really fast

**Without a Graphics Card (CPU):**
- 100,000 records load in 3-5 seconds
- Filtering takes about 500 milliseconds
- Complex analysis takes 5-15 seconds
- Still pretty fast, just not as instant

## What You Get

### Interactive Map
- Every vehicle as a colored dot
- Crashes marked with special symbols
- Crash hotspots shown as red circles
- Color-coded by speed or braking intensity
- Click anything to see details
- Automatically zooms to what you're looking at

### Charts and Graphs
- Speed distribution charts
- Crash patterns over time
- Road-by-road comparisons
- Hour-by-hour breakdowns
- Everything interactive - hover for details

### AI Insights Report
Type "AI insights" and get a complete report:
- All crash hotspots with GPS
- Braking intensity analysis
- When crashes happen most
- Which roads are most dangerous
- Where crashes will likely happen next
- Complete with actionable recommendations

## Common Problems and Solutions

### "Column(s) ['deceleration_g'] do not exist"

Don't worry! The system fixes this automatically:
- It uses your existing acceleration data
- Converts it to the right format
- Just refresh your browser and it works

### "Connection error" or "No API key"

Simple fix:
1. Go to https://console.groq.com/
2. Sign up for free
3. Create an API key
4. Paste it in the sidebar
5. Try again

### Map isn't updating

The system is smart about when to update:
- Type "Show all data" to reset everything
- Clear your filters if you're stuck
- Refresh the page if needed

### Charts aren't showing up

Easy fixes:
- Scroll down - charts appear below the map
- Click the X button to close old charts
- They stack up, so older ones might be hidden above

### Running slow

Try these:
- Make sure you're using the Parquet file (not CSV)
- Check if your graphics card is being used: run `nvidia-smi`
- For testing, use a smaller sample size
- Close other programs using your GPU

## What Makes This Different

### No Learning Curve
- Talk to it like a human
- No SQL, no Python, no formulas
- If you can type a question, you can use it
- Takes 2 minutes to learn

### It Predicts, Not Just Reports
- Doesn't just show what happened
- Tells you where crashes WILL happen
- Identifies early warning signs
- Gives you time to intervene

### Everything in One Place
- 6 different AI analysis features
- 15+ chart types
- Interactive maps
- All in one simple interface

### Built for Action
- Exact GPS coordinates for every finding
- Risk scores tell you what's urgent
- Clear, plain-English explanations
- Ready to present to anyone

## Technologies Used

**Core System:**
- Python 3.11 - the programming language
- Streamlit - makes the web interface
- RAPIDS cuDF - super-fast data processing using your graphics card

**Data Processing:**
- Pandas - backup if no graphics card
- NumPy - math operations
- Scikit-learn - clustering algorithm

**Visualizations:**
- Folium - the interactive maps
- Plotly - the interactive charts
- Streamlit-Folium - connects maps to the app

**AI Integration:**
- Groq API - really fast AI (recommended)
- OpenAI API - also works
- Google Gemini - another option

## File Structure

Everything lives in one main file:

**app.py** (180KB, 4,093 lines)
- Lines 1-500: Setup and configuration
- Lines 500-700: Natural language understanding
- Lines 700-1100: Data loading and filtering
- Lines 1100-1600: Map creation
- Lines 1600-2100: All the AI analysis functions
- Lines 2100-2700: Chart generation
- Lines 2700-3800: Command execution
- Lines 3800-4200: User interface

**Other files:**
- run.sh - starts everything
- requirements.txt - list of needed software
- README.md - detailed documentation (you're reading the GitHub version)

## Future Ideas

Things we're thinking about adding:
- Weather data integration (does rain cause more crashes?)
- Real-time live data streaming
- Mobile app for field use
- Automatic PDF report generation
- Compare multiple cities
- Share best practices

## Contributing

Want to help make this better? Great!

**Easy ways to contribute:**
- Report bugs or issues
- Suggest new features
- Improve documentation
- Share how you're using it
- Help others in discussions

**For developers:**
- Fork the repository
- Make your changes
- Test thoroughly
- Submit a pull request

## Support and Help

**Need help?**
- Check the troubleshooting section above
- Look at the detailed README.md in this folder
- Open an issue on GitHub
- The system literally answers questions - just ask it!

**Found a bug?**
- Open a GitHub issue
- Describe what happened
- Include the error message
- Tell us what you were trying to do

## License

MIT License - which means:
- Use it for free
- Modify it however you want
- Use it commercially
- Just keep the license file
- Make roads safer!

## Credits

**Created:** October 2025

**Built using:**
- RAPIDS AI for GPU speed
- Streamlit for the interface
- Groq for understanding natural language
- DBSCAN clustering for finding hotspots
- Love for making roads safer

## The Bottom Line

This isn't just a data visualization tool. It's an AI assistant that:
- Understands traffic safety
- Predicts crashes before they happen
- Explains everything in plain English
- Helps you save lives



Remember: The best way to learn is to try it. Get your free Groq API key, upload some data, and start asking questions. You'll be finding insights in minutes.
