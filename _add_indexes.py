import sqlite3
conn = sqlite3.connect('c:/Users/Sager/OneDrive/Desktop/portfolio_app/mobile-migration/dev_portfolio.db')
c = conn.cursor()

# Check indexes
c.execute("SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name='news_articles'")
for row in c.fetchall():
    print("INDEX:", row[0])

# Check table schema
c.execute("SELECT sql FROM sqlite_master WHERE type='table' AND tbl_name='news_articles'")
for row in c.fetchall():
    print("TABLE:", row[0])

# Add composite index for common queries
c.execute("CREATE INDEX IF NOT EXISTS ix_news_category_date ON news_articles(category, published_at DESC)")
c.execute("CREATE INDEX IF NOT EXISTS ix_news_symbols ON news_articles(related_symbols)")
conn.commit()
print("\nIndexes created!")

# Verify
c.execute("SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name='news_articles'")
for row in c.fetchall():
    print("INDEX:", row[0])

conn.close()
