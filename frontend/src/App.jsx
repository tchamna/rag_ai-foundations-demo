import React, { useState, useEffect, useRef } from 'react'
import axios from 'axios'

export default function App(){
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [threshold, setThreshold] = useState(0.25)
  const [useChatGPT, setUseChatGPT] = useState(false)

  const [uploadStatus, setUploadStatus] = useState('')
  const fileInputRef = useRef()
  const messagesRef = useRef()

  useEffect(()=>{ if(messagesRef.current){ messagesRef.current.scrollTop = messagesRef.current.scrollHeight } }, [messages])

  async function send(){
    if(!input.trim()) return
    const userMsg = {role: 'user', text: input}
    setMessages(m=>[...m, userMsg])
    setLoading(true)
    try{
      const form = new FormData()
      form.append('question', input)
      form.append('threshold', threshold)
      form.append('use_chatgpt', useChatGPT)

      const res = await axios.post('http://backend:8000/query', form)
      const assistant = {role: 'assistant', text: res.data.answer}
      setMessages(m=>[...m, assistant])
      setInput('')
    }catch(err){
      setMessages(m=>[...m, {role:'assistant', text: 'Error: '+(err.response?.data?.detail||err.message)}])
    }finally{ setLoading(false) }
  }

  async function handleUpload(e){
    const file = e.target.files[0]
    if(!file) return
    setUploadStatus('Uploading...')
    try{
      const form = new FormData()
      form.append('file', file)
      const res = await axios.post('http://backend:8000/upload', form)
      setUploadStatus('Uploaded: ' + res.data.name)
    }catch(err){
      setUploadStatus('Error: ' + (err.response?.data?.detail || err.message))
    }
    fileInputRef.current.value = ''
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <h2>RAG Chat</h2>
        <div>
          <label>Similarity threshold</label>
          <input type="range" min="0" max="1" step="0.01" value={threshold} onChange={e=>setThreshold(e.target.value)} />
          <div>{threshold}</div>
        </div>
        <div>
          <label><input type="checkbox" checked={useChatGPT} onChange={e=>setUseChatGPT(e.target.checked)} /> Use ChatGPT API</label>
        </div>
        <div className="controls">
          <button onClick={async ()=>{ await axios.post('http://backend:8000/rebuild'); alert('Rebuilt') }}>Rebuild Index</button>
          <a href="http://backend:8000/transcript" target="_blank">Download Transcript (CSV)</a>
        </div>
        <div style={{marginTop:20}}>
          <label>Upload document (.txt, .csv, .xlsx):</label>
          <input type="file" ref={fileInputRef} onChange={handleUpload} accept=".txt,.csv,.xlsx" />
          <div style={{fontSize:'0.9em',color:'#38bdf8'}}>{uploadStatus}</div>
        </div>
      </aside>
      <main className="chat">
        <div className="messages" ref={messagesRef}>
          {messages.map((m,i)=> (
            <div key={i} className={"msg "+m.role}>
              <div className="bubble">{m.text}</div>
            </div>
          ))}
        </div>
        <div className="composer">
          <textarea value={input} onChange={e=>setInput(e.target.value)} placeholder="Ask a question..." rows={2}></textarea>
          <button onClick={send} disabled={loading}>{loading? '...' : 'Send'}</button>
        </div>
      </main>
    </div>
  )
}
