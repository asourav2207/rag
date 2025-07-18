import streamlit as st
import sqlalchemy

def validate_redshift_secrets():
    try:
        creds = st.secrets["redshift"]
        required = ["host", "port", "database", "user", "password"]
        for k in required:
            if k not in creds or not creds[k]:
                st.error(f"Missing required Redshift secret: {k}")
                return False
        # Check for protocol or special chars in host
        host = creds["host"]
        if any(x in host for x in ["https://", "@", "*"]):
            st.error(f"Redshift host should not contain protocol, @, or *: {host}")
            return False
        # Try connecting
        engine = sqlalchemy.create_engine(
            f"postgresql+psycopg2://{creds['user']}:{creds['password']}@{creds['host']}:{creds['port']}/{creds['database']}"
        )
        with engine.connect() as conn:
            conn.execute(sqlalchemy.text("SELECT 1"))
        st.success("Redshift connection successful!")
        return True
    except Exception as e:
        st.error(f"Redshift connection failed: {e}")
        return False

if __name__ == "__main__":
    validate_redshift_secrets()
