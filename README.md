# 🎵 Discovering Your Soul Through Music

This project is a music recommendation and personality prediction system. It leverages objective Spotify audio features (e.g., energy, danceability, valence) and K-Means clustering to map musical preferences to personality traits.

## 🚀 Features

*   **User Account Management:** Secure registration, login, and account deletion functionality with password hashing (SHA-256).
*   **Music Search Engine:** Filter and search for songs based on keywords (title, artist), genres, and release year ranges. Includes direct links to YouTube search results.
*   **Personality Prediction System:**
    *   Uses **K-Means clustering** to present a diverse, sampled set of songs for users to select their preferences.
    *   Calculates the average of 7 key audio features (energy, danceability, positiveness, speechiness, liveness, acousticness, instrumentalness) from selected songs.
    *   Maps these averages to predefined personality types and generates match percentages.
*   **Data Visualization:** Interactive Radar charts and Bar charts built with Plotly to visualize user audio feature profiles and personality match scores.
*   **Soul Matching (Recommendation):** Recommends compatible personality types and curates shared "Soulmate Playlists" based on user personality and chosen social scenarios (e.g., studying, relaxing, exercising).

## 🛠️ Tech Stack

*   **Backend & Data Processing:** Python, pandas, scikit-learn (K-Means clustering)
*   **Frontend UI:** Streamlit, Plotly (Interactive Charts)
*   **Database:** MySQL (Relational Database Management System), SQLAlchemy (ORM/Connection), mysql-connector-python

## 🗄️ Database Schema

The system utilizes a 9-table normalized relational database schema (exported as `music_db_export.aql`). Key tables include:
*   **User Management:** `Users`
*   **Music Data:** `Songs`, `Genres`, `Song_Genres`
*   **Personality & Logic:** `Personality_Types`, `Personality_Result`, `Personality_Match`
*   **User Interactions:** `User_Selected_Songs`, `Recommended_Songs`

## ⚙️ Installation & Setup

Follow these steps to run the application locally.

### 1. Prerequisites
*   Python 3.8+
*   MySQL Server running locally or remotely

### 2. Clone the Repository
```bash
git clone <your-repository-url>
cd <your-repository-directory>
```

### 3. Install Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```

*(Ensure your `requirements.txt` includes: `streamlit`, `pandas`, `scikit-learn`, `sqlalchemy`, `mysql-connector-python`, `plotly`)*

### 4. Database Setup
1.  Create a MySQL database named `music_db_new`.
2.  Import the provided database dump:

    ```bash
    mysql -u root -p music_db_new < music_db_export.aql
    ```
3.  **Configure Database Connection:** Open `app.py` (or your main script file) and locate the `get_engine()` function. Update the connection string with your MySQL credentials:

    ```python
    # Update username, password, host, and database name
    return create_engine("mysql+mysqlconnector://root:YOUR_PASSWORD@localhost/music_db_new") 
    ```

### 5. Run the Application
Start the Streamlit server:

```bash
streamlit run app.py
```

The application should now be running, typically accessible at `http://localhost:8501`.
