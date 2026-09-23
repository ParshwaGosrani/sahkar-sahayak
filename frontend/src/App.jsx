import { useState } from 'react';
import { Send, User, Bot, Loader2 } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

export default function App() {
  const [messages, setMessages] = useState([
    { role: 'ai', text: 'Namaste! I am Sahkar Sahayak. How can I assist you with PMFBY and Cooperative Schemes today?', sources: [] }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userText = input;
    setMessages(prev => [...prev, { role: 'user', text: userText }]);
    setInput('');
    setLoading(true);

    try {
      const response = await fetch('http://127.0.0.1:8000/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userText, language: "en" })
      });
      
      const data = await response.json();
      setMessages(prev => [...prev, { role: 'ai', text: data.answer, sources: data.sources }]);
    } catch (error) {
      setMessages(prev => [...prev, { role: 'ai', text: 'Error connecting to the Sahkar Sahayak server.', sources: [] }]);
    }
    setLoading(false);
  };

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      <header className="bg-green-600 text-white p-4 text-center text-xl font-bold shadow-md">
        Sahkar Sahayak
      </header>
      
      <main className="flex-1 overflow-y-auto p-4 space-y-4 max-w-3xl mx-auto w-full">
        {messages.map((msg, idx) => (
          <div key={idx} className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            {msg.role === 'ai' && <div className="bg-green-100 p-2 rounded-full h-fit"><Bot size={24} className="text-green-700"/></div>}
            
            <div className={`p-4 rounded-xl max-w-[80%] shadow-sm ${msg.role === 'user' ? 'bg-blue-600 text-white' : 'bg-white border border-gray-200'}`}>
              
              {/* If it's a user message, show plain text. If AI, use Markdown. */}
              {msg.role === 'user' ? (
                <p className="whitespace-pre-wrap">{msg.text}</p>
              ) : (
                <div className="text-sm sm:text-base">
                  <ReactMarkdown 
                    components={{
                      p: ({node, ...props}) => <p className="mb-2" {...props} />,
                      ul: ({node, ...props}) => <ul className="list-disc ml-6 mb-2 space-y-1" {...props} />,
                      ol: ({node, ...props}) => <ol className="list-decimal ml-6 mb-2 space-y-1" {...props} />,
                      strong: ({node, ...props}) => <strong className="font-bold text-gray-900" {...props} />,
                      li: ({node, ...props}) => <li className="leading-relaxed" {...props} />
                    }}
                  >
                    {msg.text}
                  </ReactMarkdown>
                </div>
              )}

              {msg.sources?.length > 0 && (
                <div className="mt-3 pt-2 border-t border-gray-200 text-xs text-gray-500">
                  <strong>Sources:</strong> {msg.sources.join(', ')}
                </div>
              )}
            </div>

            {msg.role === 'user' && <div className="bg-blue-100 p-2 rounded-full h-fit"><User size={24} className="text-blue-700"/></div>}
          </div>
        ))}
        {loading && (
          <div className="flex gap-3 justify-start text-gray-500">
            <Loader2 className="animate-spin" size={24} /> Processing...
          </div>
        )}
      </main>

      <footer className="p-4 bg-white border-t border-gray-200">
        <form onSubmit={handleSend} className="max-w-3xl mx-auto flex gap-2">
          <input 
            type="text" 
            value={input} 
            onChange={e => setInput(e.target.value)} 
            placeholder="Ask a question in English, Hindi, or Marathi..."
            className="flex-1 p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500"
            disabled={loading}
          />
          <button 
            type="submit" 
            disabled={loading}
            className="bg-green-600 hover:bg-green-700 text-white p-3 rounded-lg flex items-center justify-center transition-colors"
          >
            <Send size={20} />
          </button>
        </form>
      </footer>
    </div>
  );
}