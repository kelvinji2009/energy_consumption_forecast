import React, { useState, useEffect, useRef } from 'react';
import apiClient from '../apiClient';

const TaskStatus = ({ task, onTaskComplete }) => {
    const [statusInfo, setStatusInfo] = useState(null);
    const [error, setError] = useState(null);
    const intervalRef = useRef(null);
    const taskId = task?.id;

    useEffect(() => {
        const fetchStatus = async () => {
            if (!taskId) return;
            try {
                const data = await apiClient(`/admin/tasks/${taskId}/status`);
                setStatusInfo(data);

                if (data.status === 'SUCCESS' || data.status === 'FAILURE') {
                    clearInterval(intervalRef.current);
                    if (onTaskComplete) {
                        onTaskComplete(data.status);
                    }
                }
            } catch (err) {
                setError(`Failed to fetch task status: ${err.message}`);
                clearInterval(intervalRef.current);
            }
        };

        // Clear previous interval if taskId changes
        if (intervalRef.current) {
            clearInterval(intervalRef.current);
        }

        if (taskId) {
            // Set initial status from prop, then start polling
            setStatusInfo({ task_id: taskId, status: task.status, result: null });
            intervalRef.current = setInterval(fetchStatus, 3000); // Poll every 3 seconds
        }

        // Cleanup on component unmount
        return () => {
            if (intervalRef.current) {
                clearInterval(intervalRef.current);
            }
        };
    }, [taskId, task, onTaskComplete]);

    if (!task) {
        return null;
    }

    if (error) {
        return <p style={{ color: 'red', marginTop: '1rem' }}>{error}</p>;
    }

    if (!statusInfo) {
        return <p style={{ marginTop: '1rem' }}>Awaiting task status...</p>;
    }

    let statusText = `Status: ${statusInfo.status}`;
    let resultText = '';
    let color = 'black';

    if (statusInfo.status === 'STARTED') {
        statusText = 'Status: Task is starting...';
        color = '#3f51b5';
    } else if (statusInfo.status === 'PROGRESS' && statusInfo.result?.status) {
        statusText = `Status: In Progress... (${statusInfo.result.status})`;
        color = '#3f51b5';
    } else if (statusInfo.status === 'SUCCESS') {
        statusText = 'Status: Completed Successfully!';
        resultText = statusInfo.result?.metrics ? `Metrics: ${JSON.stringify(statusInfo.result.metrics)}` : 'Done.';
        color = 'green';
    } else if (statusInfo.status === 'FAILURE') {
        statusText = 'Status: Failed';
        resultText = `Error: ${statusInfo.result?.error}`;
        color = 'red';
    }

    return (
        <div style={{ marginTop: '1rem', padding: '1rem', border: `1px solid ${color}`, borderRadius: '4px', maxWidth: '500px', margin: '1rem auto' }}>
            <p style={{ color, fontWeight: 'bold' }}>Task ID: {statusInfo.task_id}</p>
            <p style={{ color }}>{statusText}</p>
            {resultText && <p style={{ color, fontSize: '0.9rem' }}>{resultText}</p>}
        </div>
    );
};

export default TaskStatus;
