import streamlit as st

st.title('🎉 はじめてのStreamlitアプリ')
st.write('ボタンを押してみてください！')

if st.button('押してみる'):
    st.write('ボタンが押されました！')
else:
    st.write('まだ押されていません。')

