import React from 'react';
import ReactDOM from 'react-dom/client';

function RuntimePlaceholder() {
  return <div>TCM Knowledge Graph Platform</div>;
}

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    <RuntimePlaceholder />
  </React.StrictMode>,
);
