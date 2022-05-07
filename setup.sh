mkdir -p ~/.streamlit/
echo "
[general]n
email = "purushottamd.16@gmail"n
" > ~/.streamlit/credentials.toml
echo "
[server]n
headless = truen
enableCORS=falsen
port = $PORTn
" > ~/.streamlit/config.toml