import sqlite3
conn = sqlite3.connect('c:/Users/Sager/OneDrive/Desktop/portfolio_app/mobile-migration/dev_portfolio.db')
c = conn.cursor()
c.execute('DELETE FROM news_articles')
conn.commit()
remaining = c.execute("SELECT COUNT(*) FROM news_articles").fetchone()[0]
print(f"Deleted all articles. Remaining: {remaining}")
conn.close()
