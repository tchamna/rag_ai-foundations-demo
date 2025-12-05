"""
Simple script to validate if a CSV file is compatible with the RAG ingestion pipeline.
Usage: python scripts/validate_csv.py path/to/your/file.csv
"""
import sys
import pandas as pd
from pathlib import Path

def validate_csv(csv_path: str):
    """Check if CSV is compatible with the ingestion pipeline."""
    try:
        path = Path(csv_path)
        
        print(f"📄 Validating: {path.name}")
        print(f"   Path: {path.absolute()}")
        print()
        
        # Read CSV
        df = pd.read_csv(path)
        
        print(f"✅ CSV loaded successfully!")
        print(f"   Rows: {len(df)}")
        print(f"   Columns: {list(df.columns)}")
        print()
        
        # Check format
        if "question" in df.columns and "answer" in df.columns:
            print("✅ FAQ format detected (question + answer columns)")
            
            # Count valid rows
            valid_count = 0
            for idx, row in df.iterrows():
                q = str(row.get("question", "")).strip()
                a = str(row.get("answer", "")).strip()
                if q and a:
                    valid_count += 1
            
            print(f"   Valid Q&A pairs: {valid_count}/{len(df)}")
            
            # Show sample
            print()
            print("📋 Sample entries (first 3):")
            for idx, row in df.head(3).iterrows():
                q = str(row.get("question", "")).strip()
                a = str(row.get("answer", "")).strip()
                print(f"\n   Row {idx+1}:")
                print(f"   Q: {q[:100]}{'...' if len(q) > 100 else ''}")
                print(f"   A: {a[:100]}{'...' if len(a) > 100 else ''}")
        
        else:
            print("✅ Generic CSV format detected")
            print("   All column values will be concatenated into text chunks")
            print()
            print("📋 Sample row (first row):")
            if len(df) > 0:
                first_row = df.iloc[0]
                values = [str(v).strip() for v in first_row.values if str(v).strip()]
                combined = " ".join(values)
                print(f"   {combined[:200]}{'...' if len(combined) > 200 else ''}")
        
        print()
        print("=" * 60)
        print("✅ CSV IS COMPATIBLE!")
        print()
        print("Next steps:")
        print(f"1. Copy file to: data/docs/")
        print(f"   cp '{path.absolute()}' data/docs/{path.name}")
        print()
        print("2. Rebuild vector store:")
        print("   - Via app: Click 'Rebuild Vector Store' in sidebar")
        print("   - Via CLI: python src/ingest.py")
        print("=" * 60)
        
        return True
        
    except FileNotFoundError:
        print(f"❌ ERROR: File not found: {csv_path}")
        return False
    except pd.errors.EmptyDataError:
        print(f"❌ ERROR: CSV file is empty")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/validate_csv.py path/to/your/file.csv")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    success = validate_csv(csv_path)
    sys.exit(0 if success else 1)
