import { Stack } from 'expo-router';

export default function RootLayout() {
  return (
    <Stack
      screenOptions={{
        headerStyle: {
          backgroundColor: '#fff', 
        },
        headerShadowVisible: false, 
        headerTitle: "E C H I F Y",
        headerTitleStyle: {
          fontWeight: 'bold',
          color: '#6d3d1e', 
          fontSize: 20,
        },
        headerTitleAlign: 'center',
      }}
    />
  );
}