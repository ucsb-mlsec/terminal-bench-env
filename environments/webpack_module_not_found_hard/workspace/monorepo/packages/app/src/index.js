import React from 'react';
import { formatText, add } from '@myorg/utils';
import { Button } from '@myorg/components';

function App() {
  const formattedMessage = formatText('hello world');
  const sum = add(2, 3);
  
  console.log('Formatted text:', formattedMessage);
  console.log('Sum result:', sum);
  
  return React.createElement(
    'div',
    { className: 'app' },
    React.createElement('h1', null, formattedMessage),
    React.createElement('p', null, `The sum is: ${sum}`),
    React.createElement(Button, { label: 'Click me' })
  );
}

export default App;