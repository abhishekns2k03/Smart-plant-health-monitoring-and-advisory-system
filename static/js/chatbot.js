$.ajax({
    url: '/chatbot', // The endpoint for your Flask route
    type: 'POST', // Ensure this is POST
    contentType: 'application/json',
    data: JSON.stringify({ message: userInput }),
    // ...
});